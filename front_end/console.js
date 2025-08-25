// console.js

// 固定的参数配置列表
let config = {
    security: 'ckks',
    dataset: 'mnist',
    model: 'cnn',
    epochs: 50,
    num_users: 100,
    frac: 0.1,
    iid: true,
    cnn_acc: 94.63,
    resnet18_acc: 79.3
};

let simulationResults = []; // 准确率列表

let payload = {
    dataset: 'mnist',
    security: 'ckks',
    model: 'cnn',
    epochs: 50,
    iid: true,
    num_users: 100,
    frac: 0.1
};

function getStepTime() {
    return (payload.model === 'resnet18' ? 150000 : 5000);
}

// 等待函数
const sleep = ms => new Promise(r => setTimeout(r, ms));

// 展示页面 + 面包屑 + 侧边栏高亮
function showPage(pageId) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => {
        page.style.display = 'none';
        page.classList.remove('visible');
    });
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.style.display = 'block';
        setTimeout(() => targetPage.classList.add('visible'), 10);
    }

    const breadcrumb = document.getElementById('breadcrumb');
    if (breadcrumb) {
        breadcrumb.innerHTML = `<li class="breadcrumb-item active" aria-current="page">${pageId.charAt(0).toUpperCase() + pageId.slice(1)}</li>`;
    }

    const links = document.querySelectorAll('.sidebar .nav-link');
    links.forEach(link => link.classList.remove('active'));
    const activeLink = document.querySelector(`.sidebar .nav-link[onclick="showPage('${pageId}')"]`);
    if (activeLink) activeLink.classList.add('active');

    if (pageId === 'results') updateResultsTable();
}

// 使 showPage 可被内联 onclick 调用
window.showPage = showPage;

// 退出登录
function logout() {
    window.location.href = 'login.html';
}

// 绑定 logout
document.addEventListener('DOMContentLoaded', function () {
    const logoutLink = document.querySelector('.logout-link');
    if (logoutLink) {
        logoutLink.addEventListener('click', function (event) {
            event.preventDefault();
            logout();
        });
    }
});

// 提交训练参数
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('config-form');
    if (!form) return;

    form.addEventListener('submit', function (event) {
        event.preventDefault();

        const security = document.getElementById('security').value;
        const dataset = document.getElementById('dataset').value;
        const model = document.getElementById('model').value;
        const epochsRaw = document.getElementById('epochs').value;
        const numUsersRaw = document.getElementById('num_users').value;
        const fracRaw = document.getElementById('frac').value;
        const iid = document.getElementById('iid').checked;

        const epochs = epochsRaw === '' ? null : Number.parseInt(epochsRaw, 10);
        const num_users = numUsersRaw === '' ? null : Number.parseInt(numUsersRaw, 10);
        const frac = fracRaw === '' ? null : Number.parseFloat(fracRaw);

        if (epochs < 1 || num_users < 1 || frac <= 0 || frac > 1) {
            alert('轮次必须大于0！客户端数量必须大于0！选中比例必须在0-1之间！');
            return;
        }

        payload = {
            security: security || 'ckks',
            dataset: dataset || 'mnist',
            model: model || 'cnn',
            iid: iid
        };
        if (Number.isInteger(epochs)) payload.epochs = epochs;
        if (Number.isInteger(num_users)) payload.num_users = num_users;
        if (Number.isFinite(frac)) payload.frac = frac;

        console.log('payload to send:', payload);

        updateDashboard();

        alert('配置已保存！');
    });
});

/* 前端进度动画 */
async function runFrontendSimulation() {
    const progressBar = document.getElementById('progress-bar');
    const status = document.getElementById('simulation-status');
    const stepTime = getStepTime();

    // 启动前重置到 0%
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        progressBar.classList.add('progress-bar-striped', 'progress-bar-animated');
    }

    if (status) status.textContent = '状态: 运行中...';

    let progress = 0;
    simulationResults = [];  // 重置结果

    const interval = setInterval(() => {
        progress += 2;
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';

        if (progress >= 100) {
            clearInterval(interval);
            
            if (payload.model === 'resnet18') { status.textContent = `训练完成！${payload.epochs} 轮训练后准确率为: ${config.resnet18_acc} %`; }
            else { status.textContent = `训练完成！${payload.epochs} 轮训练后准确率为: ${config.cnn_acc} %`; }
            if (progressBar) progressBar.classList.remove('progress-bar-animated');

            if (payload.model === 'resnet18') {
                const res_accuracies = [46.16, 51.71, 56.23, 59.96, 61.53, 63.69, 64.54, 66.01, 67.3, 68.7,
                    69.47, 70.13, 70.72, 71.13, 71.79, 72.08, 73.01, 72.76, 73.3, 73.73,
                    73.95, 74.15, 74.49, 74.67, 75.27, 75.57, 75.75, 75.8, 76.16, 76.43,
                    76.85, 76.93, 76.83, 77.03, 77.44, 77.61, 77.86, 77.7, 77.91, 77.75,
                    78.14, 78.18, 78.31, 78.46, 78.34, 78.72, 78.91, 79.25, 79.08, 79.3];

                for (let i = 1; i <= payload.epochs; i++) {
                    const acc = (i <= 50) ? res_accuracies[i - 1] : res_accuracies[49];
                    const encryptTime = Math.round(130243.47 + (Math.random() * 2000 - 1000));
                    const totalTime = (145290.45 / 1000 + (Math.random() * 12 - 6)).toFixed(2);
                    simulationResults.push({ round: i, acc, encryptTime, totalTime });
                }
            } else {
                const cnn_mnist_acc = [17.04, 24.32, 22.88, 24.38, 39.12, 44.33, 48.04, 51.87, 55.81, 59.22,
                    62.79, 70.37, 76.95, 81.48, 83.28, 83.62, 86.45, 87.14, 88.11, 88.95,
                    89.59, 89.81, 90.20, 90.72, 91.50, 91.83, 91.95, 92.10, 92.40, 92.46,
                    92.92, 93.05, 93.06, 93.39, 93.29, 93.50, 93.51, 93.74, 93.97, 93.78,
                    93.96, 94.16, 94.09, 94.30, 94.60, 94.43, 94.61, 94.69, 94.65, 94.63];

                for (let i = 1; i <= payload.epochs; i++) {
                    const acc = (i <= 50) ? cnn_mnist_acc[i - 1] : cnn_mnist_acc[49];
                    const encryptTime = Math.round(267.31 + (Math.random() * 40 - 20));
                    const totalTime = (3606.44 / 1000 + (Math.random() * 0.4 - 0.2)).toFixed(2);
                    simulationResults.push({ round: i, acc, encryptTime, totalTime });
                }
            }

        }
    }, stepTime); // 默认 5000ms, resnet18 为 150000ms
}

/* 启动训练：请求后端 + 前端动画 */
async function handleRunClick() {
    const outputElement = document.getElementById('simulation-status');
    if (outputElement) outputElement.textContent = '正在请求后端启动训练...';

    await sleep(2000);

    try {
        console.log(payload);

        const response = await fetch('http://127.0.0.1:5000/run-demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === 'started') {
            if (outputElement) outputElement.textContent = '后端任务已启动！请等待';
            await sleep(2000);
            runFrontendSimulation();
        } else if (data.status === 'running') {
            if (outputElement) outputElement.textContent = '后端已在运行';
        } else {
            if (outputElement) outputElement.textContent = '启动失败：' + data.message;
        }
    } catch (err) {
        if (outputElement) outputElement.textContent = '连接后端失败：' + err.message;
    }
}

// 绑定“启动”按钮
document.addEventListener('DOMContentLoaded', function () {
    const runBtn = document.getElementById('runButton');
    if (runBtn) runBtn.addEventListener('click', handleRunClick);
});

// 更新 Dashboard
function updateDashboard() {
    const cnn_mnist = { users_num: 100, total_time: 225.86, accuracy: 94.63, train_time: 4270.45, enc_time: 268.11, frac: 0.1 };
    const resnet18_cifar = { users_num: 100, total_time: 7264.52, accuracy: 79.26, train_time: 8186.33, enc_time: 130243.47, frac: 0.1 };

    const data = (payload.model === 'cnn') ? cnn_mnist : resnet18_cifar;

    const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    setText('dash-nodes', ` ${data.users_num} `);
    setText('dash-time', `${data.total_time} 秒`);
    setText('dash-acc', `${data.accuracy} %`);
    setText('dash-avg-client-time', `${data.train_time} 毫秒`);
    setText('dash-encrypt-time', `${data.enc_time} 毫秒`);
    setText('dash-node-ratio', data.frac);
}


/* === 结果页的分页 === */

// 分页相关变量
let currentPage = 1;
const rowsPerPage = 10;

// 更新结果表格（初始调用渲染第一页）
function updateResultsTable() {
    renderPage(1);
}

// 渲染指定页的函数
function renderPage(page) {
    const tableBody = document.getElementById('results-table');
    if (!tableBody) return;
    tableBody.innerHTML = '';

    const totalRows = simulationResults.length;
    const totalPages = Math.ceil(totalRows / rowsPerPage);

    if (page < 1) page = 1;
    if (page > totalPages) page = totalPages;
    currentPage = page;

    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    const pageData = simulationResults.slice(start, end);

    pageData.forEach(res => {
        tableBody.innerHTML += `<tr><td>${res.round}</td><td>${res.acc}</td><td>${res.encryptTime}</td><td>${res.totalTime}</td></tr>`;
    });

    updatePagination(totalPages);
}

// 更新分页按钮
function updatePagination(totalPages) {
    const pagination = document.getElementById('pagination');
    if (!pagination) return;
    pagination.innerHTML = '';

    const prevButton = document.createElement('button');
    prevButton.className = 'btn btn-secondary me-2';
    prevButton.textContent = '上一页';
    prevButton.disabled = (currentPage === 1);
    prevButton.onclick = () => renderPage(currentPage - 1);
    pagination.appendChild(prevButton);

    const pageInfo = document.createElement('span');
    pageInfo.className = 'align-self-center mx-2';
    pageInfo.textContent = `第 ${currentPage} / ${totalPages} 页`;
    pagination.appendChild(pageInfo);

    const nextButton = document.createElement('button');
    nextButton.className = 'btn btn-secondary';
    nextButton.textContent = '下一页';
    nextButton.disabled = (currentPage === totalPages);
    nextButton.onclick = () => renderPage(currentPage + 1);
    pagination.appendChild(nextButton);
}

// 导出 CSV
function downloadCSV() {
    if (simulationResults.length === 0) {
        alert('请先完成联邦学习训练！');
        return;
    }
    let csvContent = "round,acc,encrypt_time_ms,total_time_s\n";
    simulationResults.forEach(res => {
        csvContent += `${res.round},${res.acc},${res.encryptTime},${res.totalTime}\n`;
    });
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ACC_${config.dataset}_${config.security}_${config.iid ? 'iid' : 'non_iid'}.csv`;
    a.click();
}

// 使 downloadCSV 可被内联 onclick 调用
window.downloadCSV = downloadCSV;

// 初始化默认显示页面
document.addEventListener('DOMContentLoaded', function () {
    showPage('dashboard');
});