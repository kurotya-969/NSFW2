// 親密度ゲージと関係性表示の強化スクリプト

// 親密度レベルに応じた関係性ステージの取得
function getRelationshipStage(affectionLevel) {
    if (affectionLevel <= 10) {
        return "hostile";
    } else if (affectionLevel <= 25) {
        return "distant";
    } else if (affectionLevel <= 45) {
        return "cautious";
    } else if (affectionLevel <= 65) {
        return "friendly";
    } else if (affectionLevel <= 85) {
        return "warm";
    } else {
        return "close";
    }
}

// 関係性ステージの日本語表示
function getStageDisplayName(stage) {
    const stageNames = {
        "hostile": "敵対的",
        "distant": "距離を置く",
        "cautious": "慎重",
        "friendly": "友好的",
        "warm": "温かい",
        "close": "親密"
    };
    return stageNames[stage] || stage;
}

// 次のステージまでの残りポイントを計算
function getPointsToNextStage(affectionLevel) {
    if (affectionLevel <= 10) {
        return { nextStage: "distant", pointsNeeded: 11 - affectionLevel, percentage: affectionLevel / 11 * 100 };
    } else if (affectionLevel <= 25) {
        return { nextStage: "cautious", pointsNeeded: 26 - affectionLevel, percentage: (affectionLevel - 11) / 15 * 100 };
    } else if (affectionLevel <= 45) {
        return { nextStage: "friendly", pointsNeeded: 46 - affectionLevel, percentage: (affectionLevel - 26) / 20 * 100 };
    } else if (affectionLevel <= 65) {
        return { nextStage: "warm", pointsNeeded: 66 - affectionLevel, percentage: (affectionLevel - 46) / 20 * 100 };
    } else if (affectionLevel <= 85) {
        return { nextStage: "close", pointsNeeded: 86 - affectionLevel, percentage: (affectionLevel - 66) / 20 * 100 };
    } else {
        return { nextStage: "max", pointsNeeded: 0, percentage: 100 };
    }
}

// 背景装飾はHTML側で直接埋め込むため、この関数は不要になりました
// gr.HTML()で背景装飾コンテナを埋め込んでいます

// 段階変化通知を表示（強化版）
function showStageChangeNotification(oldStage, newStage) {
    console.log(`関係性ステージ変化: ${oldStage} -> ${newStage}`); // デバッグ用

    // 既存の通知を削除
    const existingNotification = document.querySelector('.stage-change-notification');
    if (existingNotification) {
        existingNotification.remove();
    }

    // 新しい通知を作成
    const notification = document.createElement('div');
    notification.className = `stage-change-notification stage-${newStage}`;

    // ステージに応じたメッセージを設定
    let message = '';
    let emoji = '';
    switch (newStage) {
        case 'distant':
            message = '麻理の警戒心が少し和らいだようだ...';
            emoji = '🌱';
            break;
        case 'cautious':
            message = '麻理はあなたに対して少し興味を持ち始めたようだ...';
            emoji = '👀';
            break;
        case 'friendly':
            message = '麻理はあなたに対して友好的な態度を見せ始めた！';
            emoji = '😊';
            break;
        case 'warm':
            message = '麻理はあなたに心を開き始めている...！';
            emoji = '💫';
            break;
        case 'close':
            message = '麻理はあなたを特別な存在として認めているようだ！';
            emoji = '💖';
            break;
        default:
            message = '麻理との関係性が変化した...';
            emoji = '✨';
    }

    // 通知内容を設定
    notification.innerHTML = `
        <div class="notification-icon">${emoji}</div>
        <div class="notification-message">${message}</div>
        <div class="notification-progress"></div>
    `;

    // チャットボックスの上部に挿入
    const gradioApp = document.querySelector('.gradio-app');
    if (gradioApp) {
        gradioApp.insertBefore(notification, gradioApp.firstChild);

        // プログレスバーのアニメーション
        const progressBar = notification.querySelector('.notification-progress');
        progressBar.style.width = '100%';

        // 一定時間後に通知を消す
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 1000);
        }, 8000);
    } else {
        console.error('gradio-app element not found');
    }

    // 効果音を再生（オプション）
    try {
        const audio = new Audio('data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAASAAAeMwAUFBQUFCIiIiIiIjAwMDAwMD09PT09PUxMTExMWFhYWFhYZmZmZmZmdHR0dHR0goKCgoKCkJCQkJCQnp6enp6erKysrKysvLy8vLy8ysrKysrK2NjY2NjY5ubm5ubm9PT09PT0//8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAQKAAAAAAAAHjOZTf9C');
        audio.volume = 0.3;
        audio.play().catch(e => console.log('Audio play failed:', e));
    } catch (e) {
        console.log('Audio not supported');
    }
}

// 関係性詳細情報を更新
function updateRelationshipDetails(affectionLevel, stage, relationshipInfo) {
    // 詳細情報コンテナを取得または作成
    let detailsContainer = document.querySelector('.relationship-details');
    if (!detailsContainer) {
        detailsContainer = document.createElement('div');
        detailsContainer.className = 'relationship-details';

        // アコーディオンの中に挿入
        const accordion = document.querySelector('.gradio-accordion');
        if (accordion) {
            const accordionContent = accordion.querySelector('.accordion-content');
            if (accordionContent) {
                accordionContent.appendChild(detailsContainer);
            }
        }
    }

    // 次のステージまでの情報を取得
    const nextStageInfo = getPointsToNextStage(affectionLevel);

    // 関係性の特徴を取得
    const traits = relationshipInfo && relationshipInfo.stage_traits ? relationshipInfo.stage_traits : {
        openness: "不明",
        trust: "不明",
        communication_style: "不明",
        emotional_expression: "不明"
    };

    // 詳細情報を更新
    detailsContainer.innerHTML = `
        <h4>現在の関係性: ${getStageDisplayName(stage)}</h4>
        <p>${relationshipInfo && relationshipInfo.description ? relationshipInfo.description : "関係性の詳細情報がありません"}</p>
        <ul>
            <li><strong>心の開き具合:</strong> ${traits.openness}</li>
            <li><strong>信頼度:</strong> ${traits.trust}</li>
            <li><strong>コミュニケーションスタイル:</strong> ${traits.communication_style}</li>
            <li><strong>感情表現:</strong> ${traits.emotional_expression}</li>
        </ul>
        ${nextStageInfo.nextStage !== "max" ? `
        <div class="next-stage-progress">
            <span>次のステージ「${getStageDisplayName(nextStageInfo.nextStage)}」まであと ${nextStageInfo.pointsNeeded} ポイント</span>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${nextStageInfo.percentage}%"></div>
            </div>
        </div>
        ` : '<div class="next-stage-progress">最高の関係性に達しています！</div>'}
    `;
}

// 親密度ゲージを外部に移動
function moveAffectionGaugeOutside() {
    // セッション情報アコーディオンを取得
    const sessionAccordion = document.querySelector('.gradio-accordion');
    if (!sessionAccordion) return;

    // 親密度ゲージを取得
    const affectionGauge = document.querySelector('.affection-gauge');
    if (!affectionGauge) return;

    // 親密度ゲージの親要素を取得
    const parentElement = affectionGauge.parentElement;
    if (!parentElement) return;

    // 親密度ゲージをアコーディオンの前に移動
    sessionAccordion.parentElement.insertBefore(affectionGauge, sessionAccordion);

    // スタイルを調整
    affectionGauge.classList.add('affection-gauge-outside');
}

// 親密度ゲージの初期化と更新
function initializeAndUpdateAffectionGauge() {
    // DOMが完全に読み込まれた後に実行
    document.addEventListener('DOMContentLoaded', function () {
        // 背景装飾はHTML側で直接埋め込まれています

        // 親密度スライダーを探して拡張
        const observer = new MutationObserver(function () {
            const affectionSlider = document.querySelector('input[data-testid="range"]');
            if (affectionSlider && !affectionSlider.classList.contains('affection-gauge-initialized')) {
                // スライダーの親要素にクラスを追加
                const sliderContainer = affectionSlider.closest('.gradio-slider');
                if (sliderContainer) {
                    sliderContainer.classList.add('affection-gauge');
                }

                // 初期化済みとしてマーク
                affectionSlider.classList.add('affection-gauge-initialized');

                // 親密度ゲージを外部に移動
                setTimeout(moveAffectionGaugeOutside, 500);

                // 値の変更を監視
                let lastValue = affectionSlider.value;
                let lastStage = getRelationshipStage(lastValue);

                // 値の変更を監視する関数
                const monitorValueChanges = () => {
                    const currentValue = parseFloat(affectionSlider.value);
                    const currentStage = getRelationshipStage(currentValue);

                    // ステージが変化した場合に通知
                    if (currentStage !== lastStage) {
                        showStageChangeNotification(lastStage, currentStage);
                        lastStage = currentStage;
                    }

                    // 関係性ステージ表示を更新
                    const stageDisplay = document.querySelector('.relationship-stage');
                    if (stageDisplay) {
                        // 以前のステージクラスを削除
                        stageDisplay.className = 'relationship-stage';
                        // 新しいステージクラスを追加
                        stageDisplay.classList.add(`stage-${currentStage}`);
                        stageDisplay.textContent = getStageDisplayName(currentStage);
                    }

                    // 関係性詳細情報を更新
                    if (window.currentRelationshipInfo) {
                        updateRelationshipDetails(currentValue, currentStage, window.currentRelationshipInfo);
                    }

                    lastValue = currentValue;
                };

                // 値の変更を定期的に確認
                setInterval(monitorValueChanges, 500);

                // 初期状態を設定
                monitorValueChanges();
            }
        });

        // DOM変更の監視を開始
        observer.observe(document.body, { childList: true, subtree: true });
    });
}

// 関係性ステージ表示の作成
function createRelationshipStageDisplay() {
    document.addEventListener('DOMContentLoaded', function () {
        const observer = new MutationObserver(function () {
            const affectionSlider = document.querySelector('input[data-testid="range"]');
            if (affectionSlider && !document.querySelector('.relationship-stage')) {
                const sliderContainer = affectionSlider.closest('.gradio-slider');
                if (sliderContainer) {
                    const stageDisplay = document.createElement('div');
                    stageDisplay.className = 'relationship-stage';
                    stageDisplay.textContent = getStageDisplayName(getRelationshipStage(affectionSlider.value));
                    sliderContainer.parentNode.insertBefore(stageDisplay, sliderContainer.nextSibling);
                }
            }
        });

        observer.observe(document.body, { childList: true, subtree: true });
    });
}

// グローバル変数の初期化
window.currentRelationshipInfo = null;

// 関係性情報の更新を受け取るグローバル関数
function updateRelationshipInfo(info) {
    window.currentRelationshipInfo = info;
}

// 初期化
initializeAndUpdateAffectionGauge();
createRelationshipStageDisplay();