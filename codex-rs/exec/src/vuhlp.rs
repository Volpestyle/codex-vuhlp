use crate::cli::VuhlpCli;
use codex_common::oss::ensure_oss_provider_ready;
use codex_common::oss::get_default_model_for_oss_provider;
use codex_common::oss::ollama_chat_deprecation_notice;
use codex_core::AuthManager;
use codex_core::NewThread;
use codex_core::ThreadManager;
use codex_core::auth::enforce_login_restrictions;
use codex_core::config::Config;
use codex_core::config::ConfigOverrides;
use codex_core::config::find_codex_home;
use codex_core::config::load_config_as_toml_with_cli_overrides;
use codex_core::default_client::set_default_originator;
use codex_core::git_info::get_git_repo_root;
use codex_core::models_manager::manager::RefreshStrategy;
use codex_core::protocol::AskForApproval;
use codex_core::protocol::EventMsg;
use codex_core::protocol::Op;
use codex_core::protocol::TurnAbortReason;
use codex_core::protocol::TurnCompleteEvent;
use codex_core::protocol::TurnAbortedEvent;
use codex_core::protocol::TokenUsage;
use codex_core::protocol::TokenUsageInfo;
use codex_core::protocol::TokenCountEvent;
use codex_core::protocol::StreamErrorEvent;
use codex_core::protocol::ErrorEvent;
use codex_core::protocol::AgentMessageEvent;
use codex_core::protocol::AgentReasoningEvent;
use codex_core::protocol::AgentReasoningRawContentEvent;
use codex_core::protocol::AgentMessageContentDeltaEvent;
use codex_core::protocol::ReasoningContentDeltaEvent;
use codex_core::protocol::ReasoningRawContentDeltaEvent;
use codex_core::protocol::SessionSource;
use codex_core::protocol::SandboxPolicy;
use codex_protocol::approvals::ElicitationAction;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::config_types::SandboxMode;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::user_input::UserInput;
use codex_utils_absolute_path::AbsolutePathBuf;
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tracing::error;
use tracing::info;
use tracing::warn;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum VuhlpInput {
    #[serde(rename = "user")]
    User { message: VuhlpMessage },
    #[serde(rename = "approval.resolved")]
    ApprovalResolved {
        #[serde(rename = "approvalId")]
        approval_id: String,
        resolution: ApprovalResolution,
    },
    #[serde(rename = "session.end")]
    SessionEnd,
}

#[derive(Debug, Deserialize)]
struct VuhlpMessage {
    role: String,
    content: Vec<VuhlpContentPart>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum VuhlpContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ApprovalStatus {
    Approved,
    Denied,
    Modified,
}

#[derive(Debug, Deserialize)]
struct ApprovalResolution {
    status: ApprovalStatus,
    #[serde(rename = "modifiedArgs")]
    modified_args: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum VuhlpOutput {
    #[serde(rename = "message.assistant.delta")]
    AssistantDelta { delta: String },
    #[serde(rename = "message.assistant.final")]
    AssistantFinal { content: String },
    #[serde(rename = "message.assistant.thinking.delta")]
    ThinkingDelta { delta: String },
    #[serde(rename = "message.assistant.thinking.final")]
    ThinkingFinal { content: String },
    #[serde(rename = "tool.proposed")]
    ToolProposed { tool: VuhlpToolCall },
    #[serde(rename = "approval.requested")]
    ApprovalRequested {
        #[serde(rename = "approvalId")]
        approval_id: String,
        tool: VuhlpToolCall,
    },
    #[serde(rename = "telemetry.usage")]
    TelemetryUsage {
        provider: &'static str,
        model: String,
        usage: VuhlpUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct VuhlpToolCall {
    id: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VuhlpUsage {
    input_tokens: i64,
    output_tokens: i64,
    total_tokens: i64,
}

#[derive(Debug, Default)]
struct TurnState {
    content: String,
    thinking: String,
    saw_final: bool,
    saw_thinking_final: bool,
}

#[derive(Debug)]
enum TurnOutcome {
    Completed,
    Shutdown,
}

pub async fn run_vuhlp(cli: VuhlpCli, codex_linux_sandbox_exe: Option<PathBuf>) -> anyhow::Result<()> {
    if let Err(err) = set_default_originator("codex_vuhlp".to_string()) {
        warn!(?err, "Failed to set codex vuhlp originator override");
    }

    let VuhlpCli {
        model: model_cli_arg,
        oss,
        oss_provider,
        config_profile,
        full_auto,
        dangerously_bypass_approvals_and_sandbox,
        cwd,
        skip_git_repo_check,
        add_dir,
        sandbox_mode: sandbox_mode_cli_arg,
        config_overrides,
    } = cli;

    let default_level = "info";
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(default_level))
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .with_filter(env_filter);

    let _ = tracing_subscriber::registry().with(fmt_layer).try_init();

    let sandbox_mode = if full_auto {
        Some(SandboxMode::WorkspaceWrite)
    } else if dangerously_bypass_approvals_and_sandbox {
        Some(SandboxMode::DangerFullAccess)
    } else {
        sandbox_mode_cli_arg.map(Into::<SandboxMode>::into)
    };

    let cli_kv_overrides = match config_overrides.parse_overrides() {
        Ok(v) => v,
        #[allow(clippy::print_stderr)]
        Err(e) => {
            eprintln!("Error parsing -c overrides: {e}");
            std::process::exit(1);
        }
    };

    let resolved_cwd = cwd.clone();
    let config_cwd = match resolved_cwd.as_deref() {
        Some(path) => AbsolutePathBuf::from_absolute_path(path.canonicalize()?)?,
        None => AbsolutePathBuf::current_dir()?,
    };

    #[allow(clippy::print_stderr)]
    let config_toml = {
        let codex_home = match find_codex_home() {
            Ok(codex_home) => codex_home,
            Err(err) => {
                eprintln!("Error finding codex home: {err}");
                std::process::exit(1);
            }
        };

        match load_config_as_toml_with_cli_overrides(&codex_home, &config_cwd, cli_kv_overrides.clone()).await {
            Ok(config_toml) => config_toml,
            Err(err) => {
                eprintln!("Error loading config.toml: {err}");
                std::process::exit(1);
            }
        }
    };

    let model_provider = if oss {
        let resolved = codex_core::config::resolve_oss_provider(
            oss_provider.as_deref(),
            &config_toml,
            config_profile.clone(),
        );

        if let Some(provider) = resolved {
            Some(provider)
        } else {
            return Err(anyhow::anyhow!(
                "No default OSS provider configured. Use --local-provider=provider or set oss_provider in config.toml"
            ));
        }
    } else {
        None
    };

    let model = if let Some(model) = model_cli_arg {
        Some(model)
    } else if oss {
        model_provider
            .as_ref()
            .and_then(|provider_id| get_default_model_for_oss_provider(provider_id))
            .map(std::borrow::ToOwned::to_owned)
    } else {
        None
    };

    let overrides = ConfigOverrides {
        model,
        review_model: None,
        config_profile,
        approval_policy: Some(AskForApproval::Never),
        sandbox_mode,
        cwd: resolved_cwd,
        model_provider: model_provider.clone(),
        codex_linux_sandbox_exe,
        base_instructions: None,
        developer_instructions: None,
        compact_prompt: None,
        include_apply_patch_tool: None,
        show_raw_agent_reasoning: oss.then_some(true),
        tools_web_search_request: None,
        additional_writable_roots: add_dir,
    };

    let config = Config::load_with_cli_overrides_and_harness_overrides(cli_kv_overrides, overrides).await?;

    if let Err(err) = enforce_login_restrictions(&config) {
        eprintln!("{err}");
        std::process::exit(1);
    }

    match ollama_chat_deprecation_notice(&config).await {
        Ok(Some(notice)) => {
            warn!(?notice, "Ollama wire API deprecation notice");
        }
        Ok(None) => {}
        Err(err) => {
            warn!(?err, "Failed to detect Ollama wire API");
        }
    }

    if oss {
        let provider_id = match model_provider.as_ref() {
            Some(id) => id,
            None => {
                error!("OSS provider unexpectedly not set when oss flag is used");
                return Err(anyhow::anyhow!("OSS provider not set but oss flag was used"));
            }
        };
        ensure_oss_provider_ready(provider_id, &config)
            .await
            .map_err(|e| anyhow::anyhow!("OSS setup failed: {e}"))?;
    }

    let default_cwd = config.cwd.to_path_buf();
    let default_approval_policy = config.approval_policy.value();
    let default_sandbox_policy = config.sandbox_policy.get();
    let default_effort = config.model_reasoning_effort;
    let default_summary = config.model_reasoning_summary;

    if !skip_git_repo_check && get_git_repo_root(&default_cwd).is_none() {
        eprintln!("Not inside a trusted directory and --skip-git-repo-check was not specified.");
        std::process::exit(1);
    }

    let auth_manager = AuthManager::shared(
        config.codex_home.clone(),
        true,
        config.cli_auth_credentials_store_mode,
    );
    let thread_manager = ThreadManager::new(
        config.codex_home.clone(),
        auth_manager,
        SessionSource::Exec,
    );
    let default_model = thread_manager
        .get_models_manager()
        .get_default_model(&config.model, &config, RefreshStrategy::OnlineIfUncached)
        .await;

    let mut thread = start_thread(&thread_manager, &config).await?;

    info!("codex vuhlp mode ready");

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_reset_command(trimmed) {
            info!("resetting codex thread");
            if let Err(err) = shutdown_thread(&thread).await {
                warn!(?err, "failed to shutdown codex thread during reset");
            }
            thread = start_thread(&thread_manager, &config).await?;
            continue;
        }

        let input = match serde_json::from_str::<VuhlpInput>(trimmed) {
            Ok(input) => input,
            Err(err) => {
                error!(line_len = trimmed.len(), ?err, "invalid JSONL input");
                emit_error_final(format!("Invalid JSON input: {err}"));
                emit_output(VuhlpOutput::MessageStop);
                continue;
            }
        };

        match input {
            VuhlpInput::User { message } => {
                if message.role != "user" {
                    warn!(role = message.role.as_str(), "unexpected message role; treating as user");
                }
                let Some(prompt) = extract_prompt(&message) else {
                    warn!(
                        content_len = message.content.len(),
                        "user message missing text content"
                    );
                    emit_error_final("Error: user message missing text content".to_string());
                    emit_output(VuhlpOutput::MessageStop);
                    continue;
                };

                let prompt_len = prompt.len();
                info!(prompt_len, "received prompt");

                let outcome = match run_turn(
                    &thread,
                    &mut lines,
                    prompt,
                    &default_cwd,
                    default_approval_policy,
                    &default_sandbox_policy,
                    &default_model,
                    default_effort.clone(),
                    default_summary.clone(),
                )
                .await
                {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        error!(?err, "turn failed");
                        emit_error_final(format!("Error: {err}"));
                        emit_output(VuhlpOutput::MessageStop);
                        return Err(err);
                    }
                };
                emit_output(VuhlpOutput::MessageStop);
                if matches!(outcome, TurnOutcome::Shutdown) {
                    return Ok(());
                }
            }
            VuhlpInput::ApprovalResolved { approval_id, .. } => {
                warn!(approval_id, "received approval.resolved outside of turn context, ignoring");
            }
            VuhlpInput::SessionEnd => {
                info!("received session.end, shutting down");
                break;
            }
        }
    }

    if let Err(err) = shutdown_thread(&thread).await {
        warn!(?err, "failed to shutdown codex thread on EOF");
    }

    Ok(())
}

async fn start_thread(
    thread_manager: &ThreadManager,
    config: &Config,
) -> anyhow::Result<Arc<codex_core::CodexThread>> {
    let NewThread {
        thread,
        session_configured: _,
        thread_id: _,
    } = thread_manager.start_thread(config.clone()).await?;
    Ok(thread)
}

async fn shutdown_thread(thread: &Arc<codex_core::CodexThread>) -> anyhow::Result<()> {
    let shutdown_sent = thread.submit(Op::Shutdown).await;
    if shutdown_sent.is_err() {
        return Ok(());
    }

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Err(anyhow::anyhow!("timeout waiting for shutdown"));
        }
        let remaining = deadline - now;
        let event = match tokio::time::timeout(remaining, thread.next_event()).await {
            Ok(Ok(event)) => event,
            Ok(Err(err)) => return Err(anyhow::anyhow!("shutdown event error: {err}")),
            Err(_) => return Err(anyhow::anyhow!("timeout waiting for shutdown")),
        };
        if matches!(event.msg, EventMsg::ShutdownComplete) {
            return Ok(());
        }
    }
}

async fn run_turn(
    thread: &Arc<codex_core::CodexThread>,
    lines: &mut tokio::io::Lines<BufReader<tokio::io::Stdin>>,
    prompt: String,
    cwd: &PathBuf,
    approval_policy: AskForApproval,
    sandbox_policy: &SandboxPolicy,
    model: &str,
    effort: Option<ReasoningEffortConfig>,
    summary: ReasoningSummaryConfig,
) -> anyhow::Result<TurnOutcome> {
    let mut state = TurnState::default();
    let mut last_usage: Option<TokenUsage> = None;
    let mut error_message: Option<String> = None;
    let mut last_agent_message: Option<String> = None;

    thread
        .submit(Op::UserTurn {
            items: vec![UserInput::Text {
                text: prompt,
                text_elements: Vec::new(),
            }],
            cwd: cwd.clone(),
            approval_policy,
            sandbox_policy: sandbox_policy.clone(),
            model: model.to_string(),
            effort,
            summary,
            final_output_json_schema: None,
            collaboration_mode: None,
        })
        .await?;

    loop {
        let event = thread.next_event().await?;
        match event.msg {
            EventMsg::AgentMessageContentDelta(AgentMessageContentDeltaEvent { delta, .. }) => {
                state.content.push_str(&delta);
                emit_output(VuhlpOutput::AssistantDelta { delta });
            }
            EventMsg::ReasoningContentDelta(ReasoningContentDeltaEvent { delta, .. })
            | EventMsg::ReasoningRawContentDelta(ReasoningRawContentDeltaEvent { delta, .. }) => {
                state.thinking.push_str(&delta);
                emit_output(VuhlpOutput::ThinkingDelta { delta });
            }
            EventMsg::AgentMessage(AgentMessageEvent { message }) => {
                state.content = message.clone();
                if !state.saw_final {
                    emit_output(VuhlpOutput::AssistantFinal { content: message });
                    state.saw_final = true;
                }
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text })
            | EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent { text }) => {
                state.thinking = text.clone();
                if !state.saw_thinking_final {
                    emit_output(VuhlpOutput::ThinkingFinal { content: text });
                    state.saw_thinking_final = true;
                }
            }
            EventMsg::TokenCount(TokenCountEvent { info, .. }) => {
                if let Some(TokenUsageInfo { last_token_usage, .. }) = info {
                    last_usage = Some(last_token_usage);
                }
            }
            EventMsg::TurnComplete(TurnCompleteEvent { last_agent_message: last }) => {
                last_agent_message = last;
                break;
            }
            EventMsg::TurnAborted(TurnAbortedEvent { reason }) => {
                let reason_text = match reason {
                    TurnAbortReason::Interrupted => "turn interrupted",
                    TurnAbortReason::Replaced => "turn replaced",
                    TurnAbortReason::ReviewEnded => "review ended",
                };
                error_message = Some(reason_text.to_string());
                break;
            }
            EventMsg::Error(ErrorEvent { message, .. }) => {
                error_message = Some(message);
            }
            EventMsg::StreamError(StreamErrorEvent { message, additional_details, .. }) => {
                let details = additional_details.unwrap_or_default();
                if details.is_empty() {
                    error_message = Some(message);
                } else {
                    error_message = Some(format!("{message} ({details})"));
                }
            }
            EventMsg::ElicitationRequest(ev) => {
                let approval_id = format!("{}:{:?}", ev.server_name, ev.id);
                let tool = VuhlpToolCall {
                    id: approval_id.clone(),
                    name: ev.server_name.clone(),
                    args: Some(serde_json::json!({ "message": ev.message })),
                };

                // Emit tool.proposed for visibility
                emit_output(VuhlpOutput::ToolProposed { tool: tool.clone() });

                // Emit approval.requested and wait for resolution
                emit_output(VuhlpOutput::ApprovalRequested {
                    approval_id: approval_id.clone(),
                    tool,
                });

                // Wait for approval.resolved from stdin
                let decision = loop {
                    let line = match lines.next_line().await? {
                        Some(line) => line,
                        None => {
                            info!("EOF while waiting for approval, cancelling");
                            break ElicitationAction::Cancel;
                        }
                    };

                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    match serde_json::from_str::<VuhlpInput>(trimmed) {
                        Ok(VuhlpInput::ApprovalResolved { approval_id: id, resolution }) => {
                            if id == approval_id {
                                let decision = match resolution.status {
                                    ApprovalStatus::Approved => ElicitationAction::Accept,
                                    ApprovalStatus::Denied => ElicitationAction::Decline,
                                    ApprovalStatus::Modified => {
                                        if resolution.modified_args.is_some() {
                                            warn!(
                                                approval_id = id.as_str(),
                                                "modified approval received; modifiedArgs ignored"
                                            );
                                        } else {
                                            warn!(
                                                approval_id = id.as_str(),
                                                "modified approval received without modifiedArgs"
                                            );
                                        }
                                        ElicitationAction::Accept
                                    }
                                };
                                break decision;
                            } else {
                                warn!(expected = approval_id, received = id, "approval_id mismatch");
                            }
                        }
                        Ok(VuhlpInput::SessionEnd) => {
                            info!("session.end while waiting for approval, cancelling");
                            break ElicitationAction::Cancel;
                        }
                        Ok(VuhlpInput::User { .. }) => {
                            warn!("user input received while awaiting approval");
                        }
                        Err(err) => {
                            error!(?err, "invalid JSON while waiting for approval");
                        }
                    }
                };

                thread
                    .submit(Op::ResolveElicitation {
                        server_name: ev.server_name,
                        request_id: ev.id,
                        decision,
                    })
                    .await?;
            }
            EventMsg::ShutdownComplete => {
                return Ok(TurnOutcome::Shutdown);
            }
            _ => {}
        }
    }

    if !state.saw_thinking_final && !state.thinking.is_empty() {
        emit_output(VuhlpOutput::ThinkingFinal {
            content: state.thinking.clone(),
        });
        state.saw_thinking_final = true;
    }

    if !state.saw_final {
        let content = if let Some(error_message) = error_message {
            format!("Error: {error_message}")
        } else if let Some(last_message) = last_agent_message {
            last_message
        } else if !state.content.is_empty() {
            state.content.clone()
        } else {
            "No response from Codex.".to_string()
        };
        emit_output(VuhlpOutput::AssistantFinal { content });
        state.saw_final = true;
    }

    if let Some(usage) = last_usage {
        emit_output(VuhlpOutput::TelemetryUsage {
            provider: "codex",
            model: model.to_string(),
            usage: VuhlpUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens + usage.reasoning_output_tokens,
                total_tokens: usage.total_tokens,
            },
        });
    }

    Ok(TurnOutcome::Completed)
}

#[allow(clippy::print_stdout)]
fn emit_output(output: VuhlpOutput) {
    match serde_json::to_string(&output) {
        Ok(line) => {
            println!("{line}");
        }
        Err(err) => {
            error!(?err, "failed to serialize vuhlp output event");
        }
    }
}

fn emit_error_final(message: String) {
    emit_output(VuhlpOutput::AssistantFinal { content: message });
}

fn extract_prompt(message: &VuhlpMessage) -> Option<String> {
    let mut prompt = String::new();
    for part in &message.content {
        if let VuhlpContentPart::Text { text } = part {
            prompt.push_str(text);
        }
    }
    if prompt.is_empty() {
        None
    } else {
        Some(prompt)
    }
}

fn is_reset_command(line: &str) -> bool {
    line == "/new" || line == "/clear"
}
