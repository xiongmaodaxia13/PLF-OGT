#!/usr/bin/env python
"""把全时距 CI 填入表5 (GMUICU 判别表) 的 AUROC/AUPRC 单元格.

读取 results/v4/allhorizon_ci.json, 在表5 (含 '可评价时间点' 且 'AUROC' 的表) 的
每个时距×设置行的 AUROC(列4)/AUPRC(列5) 填入 '点估计 (lo–hi)' 格式.

用法: python scripts/fill_table5_allhorizon_ci.py [docx路径]
"""
from __future__ import annotations
import sys, json, re, copy
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

REPO = Path(__file__).resolve().parents[1]
CI_JSON = REPO / "results/v4/allhorizon_ci.json"


def ctext(tc):
    return ''.join(t.text or '' for t in tc.iter(f'{W}t'))


def set_cell_text(tc, newtext):
    p = tc.find(f'{W}p')
    runs = p.findall(f'{W}r')
    if not runs:
        r = ET.SubElement(p, f'{W}r'); t = ET.SubElement(r, f'{W}t'); t.text = newtext; return
    r0 = runs[0]; ts = r0.findall(f'{W}t')
    ts[0].text = newtext
    for t in ts[1:]: t.text = ''
    for r in runs[1:]:
        for t in r.findall(f'{W}t'): t.text = ''


def main():
    docx = sys.argv[1] if len(sys.argv) > 1 else str(REPO.parent / "manuscripts/gmuicu/PLF_OGT_0726-陆老师审核版2_第一档修改.docx")
    docx = Path(docx)
    if not docx.exists():
        # 回退到微信路径
        docx = Path(r"E:/xwechat_files/wxid_andty75inb3o22_e6f7/msg/file/2026-07/PLF_OGT_0726-陆老师审核版2_第一档修改.docx")
    print(f"目标 docx: {docx}", flush=True)

    ci = json.loads(CI_JSON.read_text(encoding="utf-8"))
    print("CI 数据时距:", {m: list(ci[m].keys()) for m in ci}, flush=True)

    z = zipfile.ZipFile(docx, 'r'); names = z.namelist(); data = {n: z.read(n) for n in names}; z.close()
    nsmap = dict(re.findall(r'xmlns:(\w+)="([^"]+)"', data['word/document.xml'].decode('utf-8')[:3000]))
    for pfx, uri in nsmap.items():
        if pfx != 'xml':
            try: ET.register_namespace(pfx, uri)
            except ValueError: pass

    root = ET.fromstring(data['word/document.xml'])
    body = root.find(f'{W}body'); elems = list(body)

    def fmt(h_key, setup, metric):
        """metric: 'auroc'/'auprc'"""
        mname = "TCR" if setup == "TCR" else "OLP"
        d = ci[mname][h_key]
        v = d[metric]; c = d["ci"]
        lo = c[f"{metric}_lo"]; hi = c[f"{metric}_hi"]
        return f"{v:.3f} ({lo:.3f}–{hi:.3f})"

    filled = 0
    for e in elems:
        if e.tag != f'{W}tbl': continue
        rows = e.findall(f'{W}tr')
        if not rows: continue
        hdr = [ctext(tc) for tc in rows[0].findall(f'{W}tc')]
        # 表5: 表头含 '时距' 且 'AUROC' 且 'AUPRC', 且 '可评价' (区分MIMIC表)
        if '时距' in hdr[0] and 'AUROC' in ' '.join(hdr) and '可评价' in ' '.join(hdr):
            # 确认是 GMUICU 表 (样本量 ~66k), 非 MIMIC(~43k)
            first_data = rows[1].findall(f'{W}tc')
            n_text = ctext(first_data[2]).replace(',', '')
            try:
                if int(n_text) < 50000:
                    continue  # MIMIC 表, 跳过
            except ValueError:
                continue
            print(f"\n找到 GMUICU 判别表 ({len(rows)}行), 开始填 CI", flush=True)
            i_au = hdr.index('AUROC'); i_ap = hdr.index('AUPRC')
            for r in rows[1:]:
                cells = r.findall(f'{W}tc')
                if len(cells) <= max(i_au, i_ap): continue
                h_raw = ctext(cells[0]).strip()      # '1 h'
                setup = ctext(cells[1]).strip()       # 'TCR'/'OLP'
                m = re.match(r'(\d+)\s*h', h_raw)
                if not m: continue
                h_key = f"{m.group(1)}h"
                if h_key not in ci.get(setup, {}): continue
                set_cell_text(cells[i_au], fmt(h_key, setup, 'auroc'))
                set_cell_text(cells[i_ap], fmt(h_key, setup, 'auprc'))
                filled += 2
                print(f"  {h_raw} {setup}: AUROC/AUPRC 已填", flush=True)
            break

    data['word/document.xml'] = ET.tostring(root, xml_declaration=True, encoding='UTF-8')
    with zipfile.ZipFile(docx, 'w', zipfile.ZIP_DEFLATED) as zo:
        for n in names: zo.writestr(n, data[n])
    print(f"\n完成, 共填入 {filled} 个单元格. 写回: {docx}", flush=True)


if __name__ == "__main__":
    main()
