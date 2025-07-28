# ========================================
# 📋 M1+M2+M3 系統建置逐步執行指南
# ========================================

echo "🎯 開始 M1+M2+M3 失智症分析系統建置"
echo "📅 預估時間：15-20 分鐘"
echo ""

# ========================================
# 步驟 1：環境檢查和準備
# ========================================
echo "============================================"
echo "📋 步驟 1：環境檢查和準備"
echo "============================================"
echo "🔍 檢查當前環境..."

# 檢查 Python 版本
echo "🐍 Python 版本："
python3 --version

# 檢查當前目錄
echo "📁 當前目錄："
pwd

# 檢查現有檔案
echo "📄 現有相關檔案："
ls -la *.py *.json 2>/dev/null || echo "沒有找到相關檔案"

# 檢查端口使用情況
echo "🔌 檢查端口使用情況："
netstat -ln | grep :800 || echo "8000-8009 端口都可用"

echo "✅ 環境檢查完成"
echo ""

# ========================================
# 步驟 2：建立 M3 BPSD 資料
# ========================================
echo "============================================"
echo "📋 步驟 2：建立 M3 BPSD 資料"
echo "============================================"
echo "🧠 正在建立 M3 BPSD（行為心理症狀）知識庫..."

cat > create_m3_data.py << 'EOF'
import json

def create_m3_bpsd_data():
    """建立 M3 BPSD（行為心理症狀）知識庫"""

    print("🧠 建立 M3 BPSD 知識庫...")

    m3_data = [
        {
            "chunk_id": "M3-01",
            "module_id": "M3",
            "chunk_type": "bpsd_category",
            "title": "妄想症狀",
            "content": "常見的妄想包括：被偷竊妄想（懷疑東西被偷）、被害妄想（認為有人要傷害自己）、嫉妒妄想（懷疑配偶不忠）、身體妄想（認為身體有問題）。這些症狀會影響患者的日常生活和人際關係。",
            "keywords": ["妄想", "被偷", "被害", "懷疑", "偷竊", "嫉妒", "身體"],
            "severity_indicators": {
                "輕度": ["偶爾懷疑東西被動過", "輕微的疑心"],
                "中度": ["經常指控家人偷東西", "明確的被害想法"],
                "重度": ["持續強烈的妄想", "完全不信任他人", "妄想影響所有互動"]
            },
            "management_strategies": [
                "避免直接否定妄想",
                "轉移注意力",
                "提供安全感",
                "必要時尋求醫療協助"
            ],
            "confidence_score": 0.92,
            "source": "BPSD 臨床指引"
        },
        {
            "chunk_id": "M3-02", 
            "module_id": "M3",
            "chunk_type": "bpsd_category",
            "title": "幻覺症狀",
            "content": "失智症患者可能出現視幻覺（看到不存在的人或物）、聽幻覺（聽到聲音）、觸幻覺（感覺被觸摸）。視幻覺最常見，特別是看到已故親人或陌生人。幻覺通常在傍晚或夜間更明顯。",
            "keywords": ["幻覺", "視幻覺", "聽幻覺", "觸幻覺", "看到", "聽到", "感覺", "傍晚", "夜間"],
            "severity_indicators": {
                "輕度": ["偶爾看到模糊影像", "不確定是否真實"],
                "中度": ["清楚看到或聽到不存在的事物", "會與幻覺互動"],
                "重度": ["持續且生動的幻覺", "完全相信幻覺的真實性", "嚴重影響行為"]
            },
            "management_strategies": [
                "改善照明環境",
                "檢查是否有感染或用藥問題", 
                "提供安撫和現實導向",
                "避免爭論幻覺的真實性"
            ],
            "confidence_score": 0.89,
            "source": "BPSD 臨床指引"
        },
        {
            "chunk_id": "M3-03",
            "module_id": "M3", 
            "chunk_type": "bpsd_category",
            "title": "激動與攻擊行為",
            "content": "包括言語攻擊（大聲叫罵、威脅）和身體攻擊（打人、踢人、咬人）。通常由挫折感、疼痛、環境刺激或照護方式不當引起。攻擊行為往往有特定觸發因子。",
            "keywords": ["激動", "攻擊", "暴力", "打人", "踢人", "咬人", "叫罵", "威脅", "暴躁"],
            "severity_indicators": {
                "輕度": ["偶爾提高音量", "輕微的不耐煩"],
                "中度": ["定期出現激動", "偶爾的身體攻擊", "需要勸阻"],
                "重度": ["頻繁且強烈的攻擊行為", "造成安全威脅", "需要立即介入"]
            },
            "management_strategies": [
                "識別和避免觸發因子",
                "保持冷靜和安全距離",
                "使用轉移注意力技巧",
                "評估疼痛或不適",
                "考慮藥物治療"
            ],
            "confidence_score": 0.94,
            "source": "BPSD 臨床指引"
        },
        {
            "chunk_id": "M3-04",
            "module_id": "M3",
            "chunk_type": "bpsd_category", 
            "title": "憂鬱與焦慮",
            "content": "失智症患者常出現憂鬱症狀：持續悲傷、失去興趣、睡眠問題、食慾改變。焦慮表現為過度擔心、坐立不安、重複行為。這些情緒問題會加速認知功能退化。",
            "keywords": ["憂鬱", "焦慮", "悲傷", "擔心", "坐立不安", "睡眠問題", "食慾", "重複行為"],
            "severity_indicators": {
                "輕度": ["偶爾情緒低落", "輕微擔憂"],
                "中度": ["明顯的憂鬱或焦慮症狀", "影響日常活動"],
                "重度": ["嚴重憂鬱", "自傷風險", "完全失去興趣", "極度焦慮"]
            },
            "management_strategies": [
                "提供情感支持和陪伴",
                "維持規律作息",
                "鼓勵適度活動",
                "音樂或藝術療法",
                "必要時使用抗憂鬱藥物"
            ],
            "confidence_score": 0.90,
            "source": "BPSD 臨床指引"
        },
        {
            "chunk_id": "M3-05",
            "module_id": "M3",
            "chunk_type": "bpsd_category",
            "title": "遊走與重複行為", 
            "content": "遊走行為包括無目的走動、試圖離家、重複走同一路線。重複行為如反覆收拾東西、重複詢問同樣問題、重複動作。這些行為通常反映內在需求或不安。",
            "keywords": ["遊走", "走動", "離家", "重複", "收拾", "詢問", "不安", "路線"],
            "severity_indicators": {
                "輕度": ["偶爾無目的走動", "輕微重複行為"],
                "中度": ["定期遊走行為", "明顯重複動作", "需要監督"],
                "重度": ["持續且危險的遊走", "極度重複行為", "嚴重安全風險"]
            },
            "management_strategies": [
                "確保環境安全",
                "提供結構化活動",
                "滿足基本需求（飢餓、如廁）",
                "使用追蹤設備",
                "轉移注意力到安全活動"
            ],
            "confidence_score": 0.88,
            "source": "BPSD 臨床指引"
        },
        {
            "chunk_id": "M3-06",
            "module_id": "M3",
            "chunk_type": "bpsd_category",
            "title": "睡眠障礙與日夜顛倒",
            "content": "包括夜間失眠、白天過度嗜睡、日夜節律紊亂、夜間激動（日落症候群）。睡眠問題會加重其他 BPSD 症狀，也增加照護者負擔。",
            "keywords": ["睡眠", "失眠", "嗜睡", "日夜顛倒", "日落症候群", "夜間", "激動"],
            "severity_indicators": {
                "輕度": ["偶爾睡眠不佳", "輕微日夜混淆"],
                "中度": ["明顯睡眠模式改變", "夜間頻繁醒來"],
                "重度": ["完全日夜顛倒", "嚴重夜間激動", "極少深度睡眠"]
            },
            "management_strategies": [
                "維持規律作息",
                "增加白天光照",
                "限制白天小睡",
                "睡前放鬆活動",
                "必要時使用睡眠藥物"
            ],
            "confidence_score": 0.91,
            "source": "BPSD 臨床指引"
        },
        {
            "chunk_id": "M3-07",
            "module_id": "M3",
            "chunk_type": "bpsd_category",
            "title": "食慾與飲食行為改變",
            "content": "可能出現食慾完全喪失、暴飲暴食、吞嚥困難、拒絕進食、吃非食物物品、忘記已經用餐。這些改變會影響營養狀態和身體健康。",
            "keywords": ["食慾", "飲食", "暴食", "拒絕進食", "吞嚥困難", "忘記用餐", "營養"],
            "severity_indicators": {
                "輕度": ["偶爾忘記用餐", "食慾略減"],
                "中度": ["明顯飲食習慣改變", "需要提醒用餐"],
                "重度": ["完全拒絕進食", "營養不良風險", "需要協助餵食"]
            },
            "management_strategies": [
                "定時用餐提醒",
                "提供喜愛食物",
                "簡化用餐環境",
                "監測體重變化",
                "評估吞嚥功能"
            ],
            "confidence_score": 0.87,
            "source": "BPSD 臨床指引"
        }
    ]

    return m3_data

def save_m3_data():
    """儲存 M3 BPSD 資料"""
    m3_data = create_m3_bpsd_data()

    with open('m3_bpsd_data.json', 'w', encoding='utf-8') as f:
        json.dump(m3_data, f, ensure_ascii=False, indent=2)

    print(f"✅ M3 BPSD 資料已儲存：{len(m3_data)} 個知識片段")

    # 顯示摘要
    print("\n📊 M3 BPSD 模組摘要：")
    for item in m3_data:
        print(f"   {item['chunk_id']}: {item['title']}")

    return m3_data

if __name__ == "__main__":
    print("🧠 建立 M3 BPSD 模組...")
    save_m3_data()
    print("🎉 M3 模組建立完成！")
EOF

echo "📄 M3 資料建立腳本已創建"
echo "▶️  執行 M3 資料建立..."
python3 create_m3_data.py

echo "✅ 步驟 2 完成：M3 BPSD 資料建立成功"
echo ""

# ========================================
# 步驟 3：建立整合分析引擎
# ========================================
echo "============================================"
echo "📋 步驟 3：建立 M1+M2+M3 整合分析引擎"
echo "============================================"
echo "⚙️  正在建立整合分析引擎..."

cat > m1_m2_m3_integrated_rag.py << 'EOF'
# 簡化版 M1+M2+M3 整合引擎
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

class AnalysisResult:
    """分析結果資料結構"""
    def __init__(self):
        self.matched_codes = []
        self.symptom_titles = []
        self.confidence_levels = []
        self.bpsd_analysis = None
        self.stage_detection = None
        self.comprehensive_summary = ""
        self.action_suggestions = []
        self.retrieved_chunks = []
        self.modules_used = []

class M1M2M3IntegratedEngine:
    """M1+M2+M3 整合分析引擎"""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chunks = []
        self.vocabulary = set()

        print("🚀 初始化 M1+M2+M3 整合引擎...")
        self._load_all_modules()
        self._build_search_index()
        print("✅ M1+M2+M3 整合引擎初始化完成")

    def _load_all_modules(self):
        """載入所有模組資料"""

        # 載入 M1 資料
        print("📊 載入 M1 資料...")
        m1_data = self._create_m1_data()
        self.chunks.extend(m1_data)
        print(f"✅ M1 載入：{len(m1_data)} 個知識片段")

        # 載入 M2 資料
        print("📊 載入 M2 資料...")
        try:
            with open('m2_stage_data.json', 'r', encoding='utf-8') as f:
                m2_data = json.load(f)
                self.chunks.extend(m2_data)
                print(f"✅ M2 載入：{len(m2_data)} 個知識片段")
        except FileNotFoundError:
            print("⚠️  M2 資料檔案未找到，跳過 M2 模組")

        # 載入 M3 資料  
        print("📊 載入 M3 BPSD 資料...")
        try:
            with open('m3_bpsd_data.json', 'r', encoding='utf-8') as f:
                m3_data = json.load(f)
                self.chunks.extend(m3_data)
                print(f"✅ M3 載入：{len(m3_data)} 個知識片段")
        except FileNotFoundError:
            print("⚠️  M3 資料檔案未找到")

        total_chunks = len(self.chunks)
        print(f"🎯 總計載入：{total_chunks} 個知識片段")

        # 統計各模組
        m1_count = len([c for c in self.chunks if c.get("chunk_id", "").startswith("M1")])
        m2_count = len([c for c in self.chunks if c.get("module_id") == "M2"])
        m3_count = len([c for c in self.chunks if c.get("module_id") == "M3"])

        print(f"📋 模組分布：M1({m1_count}) + M2({m2_count}) + M3({m3_count}) = {total_chunks}")

    def _create_m1_data(self):
        """創建 M1 十大警訊資料"""
        return [
            {
                "chunk_id": "M1-01",
                "title": "記憶力減退影響日常生活",
                "content": "正常老化：偶爾忘記約會但事後會想起來。失智警訊：忘記剛發生的事、重要日期或事件；反覆詢問同樣事情。",
                "keywords": ["記憶", "健忘", "忘記", "重複", "日常生活"],
                "confidence_score": 0.95,
                "source": "TADA 十大警訊"
            },
            {
                "chunk_id": "M1-02", 
                "title": "無法勝任