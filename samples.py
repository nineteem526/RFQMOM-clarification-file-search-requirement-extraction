"""Fictional fixtures: no customer information or real standards clauses."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Sample:
    filename: str
    document_key: str
    kind: str
    stage: str
    version: str
    date: str
    text: str
    initial: bool = True

    @property
    def digest(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


SAMPLES = (
    Sample(
        "DEMO_RFQ_R1.txt", "DEMO_RFQ", "RFQ", "RFQ", "R1", "2026-01-10",
        "【虚构示例，不可用于工程设计】\n"
        "1. 钢构设计\n钢结构平台设计荷载为 5 kN/m²。供应商应提交结构计算书。\n\n"
        "2. 电气控制\nPLC 控制系统采用冗余电源。急停回路应独立于普通 PLC 控制逻辑。\n\n"
        "3. 范围\n钢构基础由客户提供，地脚螺栓的供货边界待澄清。",
    ),
    Sample(
        "DEMO_B12_R1.txt", "DEMO_B12", "B12", "RFQ", "R1", "2026-01-12",
        "【虚构示例，不可用于工程设计】\n"
        "1. 钢构补充\n维修平台设计荷载为 3 kN/m²。是否属于 RFQ 中的钢结构平台尚待确认。\n\n"
        "2. 电控补充\n控制柜防护等级要求为 IP54。柜内应预留 20% 的安装空间。",
    ),
    Sample(
        "DEMO_Clarification_R1.txt", "DEMO_CLARIFICATION", "澄清", "RFQ",
        "R1", "2026-01-18",
        "【虚构示例，不可用于工程设计】\n"
        "1. 钢构供货边界\n地脚螺栓由供应商供货，客户负责基础施工。\n\n"
        "2. 平台荷载\nRFQ 与 B12 的平台荷载描述不一致，需客户确认适用平台范围；"
        "本澄清未指定文件优先级。\n\n"
        "3. 电气控制\nPLC 品牌待客户批准，不得自行替换。",
    ),
    Sample(
        "DEMO_TE_MOM_R1.txt", "DEMO_TE_MOM", "MOM", "TE", "R1", "2026-02-03",
        "【虚构示例，不可用于工程设计】\n"
        "1. 钢构会议记录\n双方讨论维修平台采用 3 kN/m²，尚未取得客户书面批准。\n\n"
        "2. 电控会议记录\n控制柜防护等级拟改为 IP55，供应商须提交成本影响供客户确认。",
        initial=False,
    ),
    Sample(
        "DEMO_TE_MOM_R2.txt", "DEMO_TE_MOM", "MOM", "TE", "R2", "2026-02-10",
        "【虚构示例，不可用于工程设计】\n"
        "1. 钢构会议记录\n客户书面确认仅维修平台采用 3 kN/m²，其他钢结构平台仍按 RFQ。"
        "本记录未声明替代 RFQ 或 B12。\n\n"
        "2. 电控会议记录\nIP55 方案仍在报价审核中，尚未批准。",
        initial=False,
    ),
    Sample(
        "DEMO_DF_MOM_R1.txt", "DEMO_DF_MOM", "MOM", "DF", "R1", "2026-03-01",
        "【虚构示例，不可用于工程设计】\n"
        "1. 电气控制设计冻结\nPLC 冗余电源保持不变；柜体 IP55 方案待批准，不视为设计冻结。\n\n"
        "2. 钢构设计冻结\n结构计算书提交后方可确认平台梁截面。",
        initial=False,
    ),
)

ALIASES = {
    "钢构": ("钢构", "钢结构", "平台", "结构计算书", "地脚螺栓"),
    "钢结构": ("钢构", "钢结构", "平台", "结构计算书", "地脚螺栓"),
    "电气控制": ("电气", "电控", "PLC", "控制柜", "急停", "冗余电源"),
    "电控": ("电气", "电控", "PLC", "控制柜", "急停", "冗余电源"),
    "plc": ("PLC", "控制系统", "冗余电源"),
    "荷载": ("荷载", "载荷"),
    "防护等级": ("防护等级", "IP54", "IP55"),
}
