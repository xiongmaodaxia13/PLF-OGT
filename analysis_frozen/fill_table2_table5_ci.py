#!/usr/bin/env python
"""把 #5 轨迹MAE CI 填入表2 (TCR行), #8 差值CI 填入表5 (TCR行Δ列).

读取:
  results/v4/traj_mae_ci.json        (#5, TCR 轨迹 MAE CI)
  results/v4/ot_paired_allhorizon.json (#8, OLP→TCR 差值 CI)

表2 (idx含'SOFA总分MAE'): TCR行 SOFA总分MAE(列2)/宏平均MAE(列3) 填CI
表5 (idx含'可评价'&'AUROC', GMUICU): TCR行 ΔAUROC(列6)/ΔAUPRC(列7) 填CI

用法: python scripts/fill_table2_table5_ci.py [docx]
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
REPO = Path(__file__).resolve().parents[1]


def ctext(tc): return ''.join(t.text or '' for t in tc.iter(f'{W}t'))


def set_cell_text(tc, newtext):
    p = tc.find(f'{W}p'); runs = p.findall(f'{W}r')
    if not runs:
        r = ET.SubElement(p, f'{W}r'); t = ET.SubElement(r, f'{W}t'); t.text = newtext; return
    r0 = runs[0]; ts = r0.findall(f'{W}t'); ts[0].text = newtext
    for t in ts[1:]: t.text = ''
    for r in runs[1:]:
        for t in r.findall(f'{W}t'): t.text = ''


def main():
    docx = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"E:/xwechat_files/wxid_andty75inb3o22_e6f7/msg/file/2026-07/PLF_OGT_0726-陆老师审核版2_第一档修改.docx")
    print(f"目标: {docx}", flush=True)

    traj = json.loads((REPO / "results/v4/traj_mae_ci.json").read_text(encoding="utf-8"))
    otp = json.loads((REPO / "results/v4/ot_paired_allhorizon.json").read_text(encoding="utf-8"))

    z = zipfile.ZipFile(docx, 'r'); names = z.namelist(); data = {n: z.read(n) for n in names}; z.close()
    nsmap = dict(re.findall(r'xmlns:(\w+)="([^"]+)"', data['word/document.xml'].decode('utf-8')[:3000]))
    for pfx, uri in nsmap.items():
        if pfx != 'xml':
            try: ET.register_namespace(pfx, uri)
            except ValueError: pass
    root = ET.fromstring(data['word/document.xml']); body = root.find(f'{W}body'); elems = list(body)

    filled = 0
    for e in elems:
        if e.tag != f'{W}tbl': continue
        rows = e.findall(f'{W}tr')
        if not rows: continue
        hdr = [ctext(tc) for tc in rows[0].findall(f'{W}tc')]
        hjoin = ' '.join(hdr)

        # ---- 表2: 轨迹 MAE (TCR行 填 SOFA总分MAE/宏平均MAE 的CI) ----
        if 'SOFA总分MAE' in hjoin:
            i_so = hdr.index('SOFA总分MAE'); i_ma = hdr.index('器官宏平均MAE')
            for r in rows[1:]:
                cells = r.findall(f'{W}tc')
                if len(cells) <= max(i_so, i_ma): continue
                if ctext(cells[1]).strip() != 'TCR': continue
                mh = re.match(r'(\d+)\s*h', ctext(cells[0]).strip())
                if not mh: continue
                hk = f"{mh.group(1)}h"
                if hk not in traj: continue
                d = traj[hk]
                set_cell_text(cells[i_so], f"{d['sofa_total_mae']:.3f} ({d['sofa_total_ci'][0]:.3f}–{d['sofa_total_ci'][1]:.3f})")
                set_cell_text(cells[i_ma], f"{d['macro_mae']:.3f} ({d['macro_ci'][0]:.3f}–{d['macro_ci'][1]:.3f})")
                filled += 2
                print(f"  表2 {hk} TCR: 总分/宏平均 MAE 已填CI", flush=True)

        # ---- 表5: 判别 Δ列 (TCR行 填 ΔAUROC/ΔAUPRC 的CI) ----
        elif '可评价' in hjoin and 'AUROC' in hjoin and 'TCR−OLP ΔAUROC' in hjoin:
            # 确认 GMUICU 表
            fd = rows[1].findall(f'{W}tc')
            try:
                if int(ctext(fd[2]).replace(',', '')) < 50000: continue  # MIMIC表跳过
            except ValueError:
                continue
            i_dau = hdr.index('TCR−OLP ΔAUROC'); i_dap = hdr.index('TCR−OLP ΔAUPRC')
            for r in rows[1:]:
                cells = r.findall(f'{W}tc')
                if len(cells) <= max(i_dau, i_dap): continue
                if ctext(cells[1]).strip() != 'TCR': continue
                mh = re.match(r'(\d+)\s*h', ctext(cells[0]).strip())
                if not mh: continue
                hk = f"{mh.group(1)}h"
                if hk not in otp: continue
                d = otp[hk]
                set_cell_text(cells[i_dau], f"{d['delta_auroc']:+.3f} ({d['delta_auroc_lo']:.3f}–{d['delta_auroc_hi']:.3f})")
                set_cell_text(cells[i_dap], f"{d['delta_auprc']:+.3f} ({d['delta_auprc_lo']:.3f}–{d['delta_auprc_hi']:.3f})")
                filled += 2
                print(f"  表5 {hk} TCR: ΔAUROC/ΔAUPRC 已填CI", flush=True)

    data['word/document.xml'] = ET.tostring(root, xml_declaration=True, encoding='UTF-8')
    with zipfile.ZipFile(docx, 'w', zipfile.ZIP_DEFLATED) as zo:
        for n in names: zo.writestr(n, data[n])
    print(f"\n完成, 填入 {filled} 个单元格. 写回: {docx}", flush=True)


if __name__ == "__main__":
    main()
