use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::{Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{Next, from_fn};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use chrono::{Local, Timelike};
use serde::{Deserialize, Serialize};

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

struct CachedUsage {
    rows: Arc<Vec<HourlyUsage>>,
    completed_at: Instant,
}

#[derive(Default)]
struct UsageState {
    // The asynchronous lock covers the scan so queued requests share its result.
    cached: tokio::sync::Mutex<Option<CachedUsage>>,
}

impl UsageState {
    async fn load(
        self: Arc<Self>,
        refresh: bool,
        loader: impl FnOnce() -> Result<Vec<HourlyUsage>> + Send + 'static,
    ) -> Result<Arc<Vec<HourlyUsage>>> {
        let requested_at = Instant::now();
        // Own the lock in a task: disconnecting a client must not start another
        // scan while its blocking work is still running.
        tokio::spawn(async move {
            let mut cached = self.cached.lock().await;
            if let Some(snapshot) = cached.as_ref()
                && (!refresh || snapshot.completed_at >= requested_at)
            {
                return Ok(snapshot.rows.clone());
            }
            let rows = Arc::new(
                tokio::task::spawn_blocking(loader)
                    .await
                    .context("Web usage scan failed")??,
            );
            *cached = Some(CachedUsage {
                rows: rows.clone(),
                completed_at: Instant::now(),
            });
            Ok(rows)
        })
        .await
        .context("Web usage task failed")?
    }
}

#[derive(Default, Deserialize)]
struct UsageQuery {
    #[serde(default)]
    refresh: bool,
}

fn load_usage() -> Result<Vec<HourlyUsage>> {
    create_analyzer_registry()
        .load_all_stats_parallel_scoped()
        .map(|stats| hourly_usage(&stats))
}

async fn usage(State(state): State<Arc<UsageState>>, Query(query): Query<UsageQuery>) -> Response {
    let response = match state.load(query.refresh, load_usage).await {
        Ok(rows) => Json(rows.as_ref()).into_response(),
        Err(error) => {
            eprintln!("Failed to load web usage: {error:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to load usage data",
            )
                .into_response()
        }
    };
    ([(header::CACHE_CONTROL, "no-store")], response).into_response()
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
            "/data.mjs",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("web/data.mjs"),
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
        .with_state(Arc::new(UsageState::default()))
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

    #[tokio::test]
    async fn reuses_snapshot_and_refreshes_explicitly() {
        let state = Arc::new(UsageState::default());
        let first = state
            .clone()
            .load(false, || {
                Ok(vec![HourlyUsage {
                    input_tokens: 10,
                    ..Default::default()
                }])
            })
            .await
            .unwrap();
        let cached = state
            .clone()
            .load(false, || anyhow::bail!("cached request must not scan"))
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &cached));
        assert_eq!(cached[0].input_tokens, 10);
        let refreshed = state
            .clone()
            .load(true, || {
                Ok(vec![HourlyUsage {
                    input_tokens: 20,
                    ..Default::default()
                }])
            })
            .await
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &refreshed));
        assert_eq!(refreshed[0].input_tokens, 20);
    }

    #[tokio::test]
    async fn concurrent_refreshes_share_one_scan() {
        let state = Arc::new(UsageState::default());
        state.clone().load(false, || Ok(vec![])).await.unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
        let first = tokio::spawn(state.clone().load(true, move || {
            started_tx.send(()).unwrap();
            finish_rx.blocking_recv().unwrap();
            Ok(vec![])
        }));
        started_rx.await.unwrap();
        let second = state
            .clone()
            .load(true, || anyhow::bail!("concurrent request must not scan"));
        tokio::pin!(second);
        assert!(futures::poll!(second.as_mut()).is_pending());
        finish_tx.send(()).unwrap();
        let first = first.await.unwrap().unwrap();
        let second = second.await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn disconnected_client_does_not_release_running_scan() {
        let state = Arc::new(UsageState::default());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
        let request = tokio::spawn(state.clone().load(false, move || {
            started_tx.send(()).unwrap();
            finish_rx.blocking_recv().unwrap();
            Ok(vec![])
        }));
        started_rx.await.unwrap();
        request.abort();
        assert!(request.await.err().unwrap().is_cancelled());
        let next = state
            .clone()
            .load(false, || anyhow::bail!("running scan must be reused"));
        tokio::pin!(next);
        assert!(futures::poll!(next.as_mut()).is_pending());
        finish_tx.send(()).unwrap();
        assert!(next.await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn failed_refresh_keeps_previous_snapshot_and_can_retry() {
        let state = Arc::new(UsageState::default());
        let first = state.clone().load(false, || Ok(vec![])).await.unwrap();
        let error = state
            .clone()
            .load(true, || anyhow::bail!("scan failed"))
            .await
            .err()
            .unwrap();
        assert_eq!(error.to_string(), "scan failed");
        let cached = state
            .clone()
            .load(false, || anyhow::bail!("must preserve previous data"))
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &cached));
        let retry = state.clone().load(true, || Ok(vec![])).await.unwrap();
        assert!(!Arc::ptr_eq(&first, &retry));
    }

    #[tokio::test]
    async fn rejects_remote_hosts_and_serves_embedded_modules() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app()).await });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        for host in [
            "example.com",
            "127.0.0.1.example.com",
            "localhost.example.com",
        ] {
            let response = client
                .get(format!("http://{address}/api/usage"))
                .header(header::HOST, host)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
        }
        for path in ["/", "/app.js", "/data.mjs", "/style.css", "/icon.svg"] {
            let response = client
                .get(format!("http://{address}{path}"))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(!response.text().await.unwrap().is_empty());
        }
        server.abort();
        assert!(server.await.unwrap_err().is_cancelled());
    }

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
