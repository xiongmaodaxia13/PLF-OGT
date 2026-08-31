#!/usr/bin/env python
"""把 #5 OLP 轨迹MAE CI 填入表2 OLP行, OLP→TCR 差值CI 填入表2 TCR行Δ列.

读取 results/v4/traj_mae_olp_ci.json:
  - OLP行: SOFA总分MAE(列2)/宏平均MAE(列3) 填 OLP CI
  - TCR行: Δ总分MAE(列4)/Δ宏平均MAE(列5) 填 delta CI (TCR-OLP)

用法: python scripts/fill_table2_olp_ci.py [docx]
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
    d = json.loads((REPO / "results/v4/traj_mae_olp_ci.json").read_text(encoding="utf-8"))

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
        if 'SOFA总分MAE' not in ' '.join(hdr): continue
        i_so = hdr.index('SOFA总分MAE'); i_ma = hdr.index('器官宏平均MAE')
        i_dso = hdr.index('TCR−OLP Δ总分MAE'); i_dma = hdr.index('TCR−OLP Δ宏平均MAE')
        print("找到表2, 开始填 OLP CI + Δ CI", flush=True)
        for r in rows[1:]:
            cells = r.findall(f'{W}tc')
            if len(cells) <= max(i_so, i_ma, i_dso, i_dma): continue
            mh = re.match(r'(\d+)\s*h', ctext(cells[0]).strip())
            if not mh: continue
            hk = f"{mh.group(1)}h"
            if hk not in d: continue
            setup = ctext(cells[1]).strip()
            rec = d[hk]
            if setup == 'OLP':
                set_cell_text(cells[i_so], f"{rec['olp_sofa_mae']:.3f} ({rec['olp_sofa_ci'][0]:.3f}–{rec['olp_sofa_ci'][1]:.3f})")
                set_cell_text(cells[i_ma], f"{rec['olp_macro_mae']:.3f} ({rec['olp_macro_ci'][0]:.3f}–{rec['olp_macro_ci'][1]:.3f})")
                filled += 2; print(f"  {hk} OLP: MAE CI 已填", flush=True)
            elif setup == 'TCR':
                # Δ列: delta 是 TCR-OLP (负值). CI 边界也可能负, 用带符号显示点估计, 边界带符号(因为跨0需看符号)
                dslo, dshi = rec['delta_sofa_ci']; dmlo, dmhi = rec['delta_macro_ci']
                set_cell_text(cells[i_dso], f"{rec['delta_sofa']:+.3f} ({dslo:.3f}, {dshi:.3f})")
                set_cell_text(cells[i_dma], f"{rec['delta_macro']:+.3f} ({dmlo:.3f}, {dmhi:.3f})")
                filled += 2; print(f"  {hk} TCR: Δ CI 已填", flush=True)
        break

    data['word/document.xml'] = ET.tostring(root, xml_declaration=True, encoding='UTF-8')
    with zipfile.ZipFile(docx, 'w', zipfile.ZIP_DEFLATED) as zo:
        for n in names: zo.writestr(n, data[n])
    print(f"\n完成, 填入 {filled} 个单元格. 写回: {docx}", flush=True)


if __name__ == "__main__":
    main()
