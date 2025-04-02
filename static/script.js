// 定义API URL
const API_BASE_URL = `${window.location.origin}/api`;

// DOM元素
const queryInput = document.getElementById('query-input');
const searchBtn = document.getElementById('search-btn');
const topkSelect = document.getElementById('topk-select');
const providerSelect = document.getElementById('provider-select');
const resultContainer = document.getElementById('result-container');
const loadingIndicator = document.getElementById('loading-indicator');
const answerContainer = document.getElementById('answer-container');
const answerContent = document.getElementById('answer-content');
const sourcesContainer = document.getElementById('sources-container');
const sourcesList = document.getElementById('sources-list');
const lawList = document.getElementById('law-list');
const lawLoadingIndicator = document.getElementById('law-loading-indicator');
const lawListContainer = document.getElementById('law-list-container');

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    // 检查系统初始化状态
    checkInitStatus();
    
    // 绑定搜索按钮点击事件
    searchBtn.addEventListener('click', handleSearch);
    
    // 绑定输入框回车事件
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // 绑定模型提供者选择事件
    providerSelect.addEventListener('change', handleProviderChange);
    
    // 获取当前使用的模型提供者
    getCurrentProvider();
});

// 获取当前使用的模型提供者
async function getCurrentProvider() {
    try {
        const response = await fetch(`${API_BASE_URL}/get_provider`);
        const data = await response.json();
        
        if (response.ok && data.provider) {
            providerSelect.value = data.provider;
        }
    } catch (error) {
        console.error('获取模型提供者失败:', error);
    }
}

// 处理模型提供者变更
async function handleProviderChange() {
    const provider = providerSelect.value;
    const statusContainer = document.getElementById('provider-status-container');
    
    // 如果状态容器不存在，则创建
    if (!statusContainer) {
        const container = document.createElement('div');
        container.id = 'provider-status-container';
        container.className = 'provider-status-container';
        providerSelect.parentNode.appendChild(container);
    }
    
    const container = document.getElementById('provider-status-container');
    container.innerHTML = '<div class="spinner small"></div><span>切换中...</span>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/set_provider`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ provider }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            container.className = 'provider-status-container status-success';
            container.innerHTML = '<i class="fas fa-check-circle"></i><span>切换成功</span>';
            
            // 更新法律列表，因为不同提供者可能有不同的索引
            fetchLawsList();
            
            // 3秒后隐藏状态信息
            setTimeout(() => {
                container.style.display = 'none';
            }, 3000);
        } else {
            throw new Error(data.detail || '切换失败');
        }
    } catch (error) {
        console.error('切换模型提供者失败:', error);
        const container = document.getElementById('provider-status-container');
        container.className = 'provider-status-container status-error';
        container.innerHTML = `<i class="fas fa-times-circle"></i><span>切换失败: ${error.message}</span>`;
    }
}

// 检查系统初始化状态
async function checkInitStatus() {
    try {
        lawLoadingIndicator.style.display = 'flex';
        
        // 添加系统状态提示区
        if (!document.getElementById('init-status-container')) {
            const statusContainer = document.createElement('div');
            statusContainer.id = 'init-status-container';
            statusContainer.className = 'init-status-container';
            lawListContainer.insertBefore(statusContainer, lawList);
        }
        
        const statusContainer = document.getElementById('init-status-container');
        statusContainer.innerHTML = '<div class="spinner"></div><p>正在检查系统状态...</p>';
        
        const response = await fetch(`${API_BASE_URL}/status`);
        const data = await response.json();
        
        if (data.initialized) {
            // 系统已初始化，加载法律列表
            statusContainer.className = 'init-status-container status-success';
            statusContainer.innerHTML = '<i class="fas fa-check-circle"></i><p>系统已完成初始化</p>';
            fetchLawsList();
        } else {
            // 系统未初始化，显示初始化按钮
            statusContainer.className = 'init-status-container status-warning';
            statusContainer.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <p>系统尚未初始化索引，请先初始化</p>
                <button id="init-btn" class="init-btn">开始初始化</button>
            `;
            
            // 绑定初始化按钮点击事件
            document.getElementById('init-btn').addEventListener('click', initializeSystem);
        }
    } catch (error) {
        console.error('检查初始化状态失败:', error);
        const statusContainer = document.getElementById('init-status-container');
        statusContainer.className = 'init-status-container status-error';
        statusContainer.innerHTML = '<i class="fas fa-times-circle"></i><p>无法检查系统状态</p>';
    } finally {
        lawLoadingIndicator.style.display = 'none';
    }
}

// 初始化系统
async function initializeSystem() {
    try {
        const statusContainer = document.getElementById('init-status-container');
        statusContainer.className = 'init-status-container';
        statusContainer.innerHTML = '<div class="spinner"></div><p>正在初始化系统，这可能需要几分钟...</p>';
        
        // 禁用查询按钮
        searchBtn.disabled = true;
        queryInput.disabled = true;
        
        const response = await fetch(`${API_BASE_URL}/init`);
        const data = await response.json();
        
        if (data.status === 'success') {
            statusContainer.className = 'init-status-container status-success';
            statusContainer.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <p>系统初始化完成！已处理 ${data.processed_files.length} 个法律文件</p>
            `;
            
            // 启用查询按钮
            searchBtn.disabled = false;
            queryInput.disabled = false;
            
            // 加载法律列表
            fetchLawsList();
        } else {
            throw new Error('初始化失败');
        }
    } catch (error) {
        console.error('初始化系统失败:', error);
        const statusContainer = document.getElementById('init-status-container');
        statusContainer.className = 'init-status-container status-error';
        statusContainer.innerHTML = `
            <i class="fas fa-times-circle"></i>
            <p>初始化系统失败: ${error.message}</p>
            <button id="retry-init-btn" class="init-btn">重试</button>
        `;
        
        // 启用查询按钮
        searchBtn.disabled = false;
        queryInput.disabled = false;
        
        // 绑定重试按钮点击事件
        document.getElementById('retry-init-btn').addEventListener('click', initializeSystem);
    }
}

// 获取已处理的法律列表
async function fetchLawsList() {
    try {
        lawLoadingIndicator.style.display = 'flex';
        
        const response = await fetch(`${API_BASE_URL}/laws`);
        const data = await response.json();
        
        if (data.laws && Array.isArray(data.laws)) {
            renderLawsList(data.laws);
        }
    } catch (error) {
        console.error('获取法律列表失败:', error);
        renderLawsList([]);
    } finally {
        lawLoadingIndicator.style.display = 'none';
    }
}

// 渲染法律列表
function renderLawsList(laws) {
    if (laws.length === 0) {
        lawList.innerHTML = '<li class="empty-list">暂无已索引的法律</li>';
        return;
    }
    
    lawList.innerHTML = '';
    laws.forEach(law => {
        const li = document.createElement('li');
        li.innerHTML = `<i class="fas fa-gavel"></i> ${law}`;
        lawList.appendChild(li);
    });
}

// 处理搜索请求
async function handleSearch() {
    const query = queryInput.value.trim();
    const topK = parseInt(topkSelect.value);
    
    if (!query) {
        alert('请输入查询内容');
        return;
    }
    
    // 显示加载状态
    resultContainer.style.display = 'block';
    loadingIndicator.style.display = 'flex';
    answerContainer.style.display = 'none';
    sourcesContainer.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK
            }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            renderResults(data);
        } else {
            throw new Error(data.detail || '查询处理失败');
        }
    } catch (error) {
        console.error('查询失败:', error);
        answerContent.innerHTML = `<div class="error-message">查询失败: ${error.message}</div>`;
        answerContainer.style.display = 'block';
        sourcesContainer.style.display = 'none';
    } finally {
        loadingIndicator.style.display = 'none';
    }
}

// 渲染查询结果
function renderResults(data) {
    // 渲染回答
    if (data.answer) {
        // 将换行符转换为<br>标签
        const formattedAnswer = data.answer.replace(/\n/g, '<br>');
        answerContent.innerHTML = formattedAnswer;
        answerContainer.style.display = 'block';
    } else {
        answerContainer.style.display = 'none';
    }
    
    // 渲染法律依据
    if (data.sources && data.sources.length > 0) {
        renderSources(data.sources);
        sourcesContainer.style.display = 'block';
    } else {
        sourcesContainer.style.display = 'none';
    }
}

// 渲染法律依据
function renderSources(sources) {
    sourcesList.innerHTML = '';
    
    sources.forEach(source => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        
        let headerText = '';
        if (source.metadata && Object.keys(source.metadata).length > 0) {
            const meta = source.metadata;
            if (meta.law_name) {
                headerText += `《${meta.law_name}》`;
            }
            if (meta.chapter) {
                headerText += ` ${meta.chapter}`;
            }
            if (meta.article_number) {
                headerText += ` ${meta.article_number}`;
            }
        }
        
        if (!headerText) {
            headerText = '法律条文';
        }
        
        sourceItem.innerHTML = `
            <div class="source-header">${headerText}</div>
            <div class="source-content">${source.text}</div>
        `;
        
        sourcesList.appendChild(sourceItem);
    });
}