() => {
    document.documentElement.lang = "zh-CN";
    const labelComposer = () => {
        const input = document.querySelector("#chat-input textarea");
        if (input) {
            input.setAttribute("aria-label", "项目问题");
            input.setAttribute("autocomplete", "off");
        }
        const send = document.querySelector("#chat-input .submit-button");
        if (send) {
            send.setAttribute("aria-label", "发送问题");
            send.title = "发送问题";
        }
    };
    labelComposer();
    const composer = document.querySelector("#chat-composer");
    if (composer) {
        new MutationObserver(labelComposer).observe(composer, {
            childList: true, subtree: true,
        });
    }
    const status = document.querySelector("#query-status");
    if (status) {
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
    }
    const history = document.querySelector("#history-scroll");
    if (history) {
        const updateTitles = () => {
            history.querySelectorAll("#history-list label").forEach(label => {
                const text = label.querySelector("span")?.textContent.trim();
                if (text) label.title = text;
            });
        };
        new MutationObserver(updateTitles).observe(history, {
            childList: true, subtree: true, characterData: true,
        });
        updateTitles();
    }
}
