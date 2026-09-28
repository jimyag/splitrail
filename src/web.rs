use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use anyhow::{Context, Result};
use axum::extract::Request;
use axum::http::{StatusCode, header};
use axum::middleware::{Next, from_fn};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use chrono::{Local, Timelike};
use serde::Serialize;

use crate::create_analyzer_registry;
use crate::types::{MessageRole, MultiAnalyzerStats};

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct HourlyUsage {
    date: String,
    hour: u8,
    model: String,
    project: String,
    tool: String,
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    reasoning_tokens: u64,
    cost: f64,
}

fn hourly_usage(stats: &MultiAnalyzerStats) -> Vec<HourlyUsage> {
    let mut hours: BTreeMap<(String, u8, String, String, String), HourlyUsage> = BTreeMap::new();
    for analyzer in &stats.analyzer_stats {
        for message in &analyzer.messages {
            if message.role != MessageRole::Assistant {
                continue;
            }
            let local_time = message.date.with_timezone(&Local);
            let date = local_time.format("%Y-%m-%d").to_string();
            let hour = local_time.hour() as u8;
            let model = message
                .model
                .as_deref()
                .unwrap_or("Unknown model")
                .to_string();
            let project = message
                .project_path
                .as_deref()
                .unwrap_or(&message.project_hash)
                .to_string();
            let tool = analyzer.analyzer_name.clone();
            let row = hours
                .entry((
                    date.clone(),
                    hour,
                    model.clone(),
                    project.clone(),
                    tool.clone(),
                ))
                .or_insert_with(|| HourlyUsage {
                    date,
                    hour,
                    model,
                    project,
                    tool,
                    ..Default::default()
                });
            row.input_tokens = row.input_tokens.saturating_add(message.stats.input_tokens);
            row.output_tokens = row
                .output_tokens
                .saturating_add(message.stats.output_tokens);
            row.cached_tokens = row
                .cached_tokens
                .saturating_add(message.stats.cached_tokens);
            row.reasoning_tokens = row
                .reasoning_tokens
                .saturating_add(message.stats.reasoning_tokens);
            row.cost += message.stats.cost;
        }
    }
    hours
        .into_values()
        .filter(|row| {
            row.input_tokens > 0
                || row.output_tokens > 0
                || row.cached_tokens > 0
                || row.reasoning_tokens > 0
                || row.cost > 0.0
        })
        .collect()
}

async fn usage() -> impl IntoResponse {
    match tokio::task::spawn_blocking(|| {
        let registry = create_analyzer_registry();
        registry
            .load_all_stats_parallel_scoped()
            .map(|stats| hourly_usage(&stats))
    })
    .await
    {
        Ok(Ok(rows)) => Json(rows).into_response(),
        Ok(Err(error)) => {
            eprintln!("Failed to load web usage: {error:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to load usage data",
            )
                .into_response()
        }
        Err(error) => {
            eprintln!("Web usage task failed: {error}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to load usage data",
            )
                .into_response()
        }
    }
}

async fn local_host_only(request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if host != "localhost"
        && !host.starts_with("localhost:")
        && host != "127.0.0.1"
        && !host.starts_with("127.0.0.1:")
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    next.run(request).await
}

fn app() -> Router {
    Router::new()
        .route("/", get(|| async { Html(include_str!("web/index.html")) }))
        .route(
            "/icon.svg",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "image/svg+xml")],
                    include_str!("web/icon.svg"),
                )
            }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("web/app.js"),
                )
            }),
        )
        .route(
            "/style.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("web/style.css"),
                )
            }),
        )
        .route("/api/usage", get(usage))
        .layer(from_fn(local_host_only))
}

pub async fn run(port: u16) -> Result<()> {
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .with_context(|| format!("Failed to listen on {address}"))?;
    println!("Splitrail web: http://{address}");
    axum::serve(listener, app())
        .await
        .context("Web server stopped")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgenticCodingToolStats, Application, ConversationMessage, Stats};
    use chrono::{TimeZone, Utc};

    #[test]
    fn groups_assistant_usage_by_local_hour_model_project_and_tool() {
        let message = ConversationMessage {
            application: Application::CodexCli,
            date: Utc.with_ymd_and_hms(2026, 9, 22, 12, 0, 0).unwrap(),
            project_hash: "hash".into(),
            project_path: Some("/work/project".into()),
            conversation_hash: "session".into(),
            local_hash: None,
            global_hash: "message".into(),
            model: Some("gpt-test".into()),
            stats: Stats {
                input_tokens: 10,
                output_tokens: 5,
                cached_tokens: 4,
                reasoning_tokens: 2,
                cost: 0.01,
                ..Default::default()
            },
            role: MessageRole::Assistant,
            uuid: None,
            session_name: None,
        };
        let mut other_model = message.clone();
        other_model.model = Some("other".into());
        other_model.global_hash = "other-message".into();
        let mut next_hour = message.clone();
        next_hour.date += chrono::Duration::hours(1);
        next_hour.global_hash = "next-hour".into();
        let mut user = message.clone();
        user.role = MessageRole::User;
        let mut zero_usage = message.clone();
        zero_usage.project_path = Some("/work/unused".into());
        zero_usage.stats = Stats::default();
        let stats = MultiAnalyzerStats {
            analyzer_stats: vec![AgenticCodingToolStats {
                daily_stats: BTreeMap::new(),
                num_conversations: 1,
                messages: vec![
                    message.clone(),
                    message,
                    next_hour,
                    other_model,
                    user,
                    zero_usage,
                ],
                analyzer_name: "Codex CLI".into(),
            }],
        };
        let rows = hourly_usage(&stats);
        assert_eq!(rows.len(), 3);
        let hour = Utc
            .with_ymd_and_hms(2026, 9, 22, 12, 0, 0)
            .unwrap()
            .with_timezone(&Local)
            .hour() as u8;
        let row = rows
            .iter()
            .find(|row| row.model == "gpt-test" && row.hour == hour)
            .unwrap();
        assert_eq!(row.project, "/work/project");
        assert_eq!(row.input_tokens, 20);
        assert_eq!(row.output_tokens, 10);
        assert_eq!(row.cached_tokens, 8);
        assert_eq!(row.reasoning_tokens, 4);
    }
}
