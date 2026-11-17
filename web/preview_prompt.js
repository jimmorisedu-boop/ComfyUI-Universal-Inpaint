// Submit only the target and its ancestors. Some extensions wrap queuePrompt
// without forwarding the partial-execution option; excluding downstream nodes
// makes previews independent of that option and of server version.
export function previewPrompt(prompt, targetId) {
    const output = prompt?.output;
    const target = String(targetId);
    if (!output || !Object.hasOwn(output, target)) {
        throw new Error(`Узел предпросмотра ${target} отсутствует в графе запуска.`);
    }
    const included = new Set();
    const visit = (id) => {
        if (included.has(id)) return;
        if (!Object.hasOwn(output, id)) throw new Error(`Не найден входной узел ${id}.`);
        included.add(id);
        for (const value of Object.values(output[id].inputs || {})) {
            if (Array.isArray(value) && value.length === 2 && Number.isInteger(value[1])) {
                visit(String(value[0]));
            }
        }
    };
    visit(target);
    return { ...prompt, output: Object.fromEntries(Object.entries(output).filter(([id]) => included.has(id))) };
}
