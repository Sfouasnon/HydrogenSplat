from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen

FONT = TTFont('Michroma.ttf'); GS = FONT.getGlyphSet(); CMAP = FONT.getBestCmap(); UPM = FONT['head'].unitsPerEm

def text_path(s, size, x, y, track=0.0):
    """Outlined text as one SVG path d; returns (d, advance width)."""
    sc = size / UPM; out = []; cx = 0.0
    for ch in s:
        g = CMAP[ord(ch)]; pen = SVGPathPen(GS)
        GS[g].draw(pen)
        d = pen.getCommands()
        if d:
            out.append(f'<path transform="translate({x+cx*sc:.2f} {y:.2f}) scale({sc:.5f} {-sc:.5f})" d="{d}"/>')
        cx += GS[g].width + track * UPM
    return ''.join(out), (cx - track * UPM) * sc

# ---- the mark: 4 parallax slats (the four views) + a gaussian splat crossbar (the record tally)
SLATS = [(22, -4.5, '#5CE1E6'), (36, -1.5, '#86B9F4'), (75, 1.5, '#A48CFF'), (89, 4.5, '#C88BFF')]

LIGHT = ['#0E9AA6', '#3F78D6', '#7359F0', '#A043DE']

def mark(mono=None, id='m', light=False):
    g = []
    if not mono:
        g.append(f'''<defs>
 <radialGradient id="{id}s" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#FFF4F4"/><stop offset="0.18" stop-color="#FF5566"/><stop offset="0.55" stop-color="#E0162B"/><stop offset="1" stop-color="#E0162B" stop-opacity="0"/></radialGradient>
 <linearGradient id="{id}g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".95"/><stop offset=".5" stop-color="#fff" stop-opacity=".55"/><stop offset="1" stop-color="#fff" stop-opacity=".85"/></linearGradient>
</defs>''')
    for k, (x, dy, c) in enumerate(SLATS):
        if light: c = LIGHT[k]
        if mono:
            g.append(f'<rect x="{x}" y="{22+dy}" width="9" height="76" rx="4.5" fill="{mono}"/>')
        else:
            g.append(f'<rect x="{x}" y="{22+dy}" width="9" height="76" rx="4.5" fill="{c}"/>'
                     f'<rect x="{x+2}" y="{25+dy}" width="2.2" height="70" rx="1.1" fill="url(#{id}g)" opacity=".55"/>')
    if mono:
        g.append(f'<ellipse cx="60" cy="60" rx="21" ry="7" fill="{mono}"/>')
    else:
        g.append(f'<ellipse cx="60" cy="60" rx="30" ry="12" fill="url(#{id}s)"/>'
                 f'<ellipse cx="60" cy="60" rx="7" ry="2.6" fill="#FFF7F7"/>')
    return ''.join(g)

def svg(w, h, body, vb=None):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{vb or f"0 0 {w} {h}"}">{body}</svg>\n'

open('hs-mark.svg','w').write(svg(120,120,mark()))
open('hs-mark-mono.svg','w').write(svg(120,120,mark(mono='currentColor')))

# ---- app icon, 1024: machined anodized body, lens mount, holographic ring, mark in the lens
def icon():
    b = '''<defs>
 <linearGradient id="body" x1="0" y1="0" x2="0.35" y2="1"><stop offset="0" stop-color="#2A2D33"/><stop offset=".45" stop-color="#15171B"/><stop offset="1" stop-color="#0A0B0D"/></linearGradient>
 <linearGradient id="bevel" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8A9099"/><stop offset=".12" stop-color="#3A3E45"/><stop offset=".9" stop-color="#101114"/><stop offset="1" stop-color="#4A4F57"/></linearGradient>
 <radialGradient id="well" cx=".5" cy=".42" r=".6"><stop offset="0" stop-color="#1C1F26"/><stop offset=".7" stop-color="#07080A"/><stop offset="1" stop-color="#000"/></radialGradient>
 <linearGradient id="holo" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5CE1E6"/><stop offset=".3" stop-color="#A48CFF"/><stop offset=".55" stop-color="#FF6FA8"/><stop offset=".8" stop-color="#FFD37A"/><stop offset="1" stop-color="#5CE1E6"/></linearGradient>
 <linearGradient id="ring" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#C9CDD3"/><stop offset=".5" stop-color="#5A5F68"/><stop offset="1" stop-color="#1E2025"/></linearGradient>
 <radialGradient id="glow" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#E0162B" stop-opacity=".35"/><stop offset="1" stop-color="#E0162B" stop-opacity="0"/></radialGradient>
</defs>
<rect x="100" y="100" width="824" height="824" rx="185" fill="url(#bevel)"/>
<rect x="108" y="110" width="808" height="806" rx="178" fill="url(#body)"/>'''
    # machined grooves (heat-sink texture) on the body, left and right of the lens
    for i in range(7):
        y = 250 + i*84
        b += f'<rect x="150" y="{y}" width="46" height="10" rx="5" fill="#000" opacity=".55"/><rect x="150" y="{y+10}" width="46" height="2" rx="1" fill="#6A6F77" opacity=".35"/>'
        b += f'<rect x="828" y="{y}" width="46" height="10" rx="5" fill="#000" opacity=".55"/><rect x="828" y="{y+10}" width="46" height="2" rx="1" fill="#6A6F77" opacity=".35"/>'
    b += '''<circle cx="512" cy="512" r="318" fill="url(#ring)"/>
<circle cx="512" cy="512" r="300" fill="#0C0D10"/>
<circle cx="512" cy="512" r="286" fill="none" stroke="url(#holo)" stroke-width="10" opacity=".9"/>
<circle cx="512" cy="512" r="268" fill="url(#well)"/>
<circle cx="512" cy="512" r="268" fill="none" stroke="#2B2F36" stroke-width="3"/>
<circle cx="512" cy="512" r="232" fill="none" stroke="#1B1E24" stroke-width="2"/>
<circle cx="512" cy="512" r="200" fill="url(#glow)"/>
<path d="M330 380 A220 220 0 0 1 560 300" fill="none" stroke="#fff" stroke-opacity=".10" stroke-width="18" stroke-linecap="round"/>
<g transform="translate(512 512) scale(3.3) translate(-60 -60)">''' + mark(id='ic') + '''</g>
<circle cx="826" cy="198" r="26" fill="#3A0006"/><circle cx="826" cy="198" r="18" fill="#E0162B"/><circle cx="820" cy="192" r="6" fill="#FFB3BB" opacity=".8"/>'''
    return svg(1024,1024,b)
open('hs-appicon.svg','w').write(icon())

# ---- lockups
def lockup(fg, sub, mono=False, light=False):
    t1, w1 = text_path('HYDROGEN', 30, 0, 0, 0.12)
    t2, w2 = text_path('SPLAT', 30, 0, 0, 0.12)
    x0 = 150
    gap = 22
    body = f'<g transform="translate(10 10)">{mark(mono=fg if mono else None, id="l", light=light)}</g>'
    body += f'<g fill="{fg}" transform="translate({x0} 0)">' + text_path('HYDROGEN', 30, 0, 86, 0.12)[0] + '</g>'
    body += f'<g fill="#E0162B" transform="translate({x0 + w1 + gap} 0)">' + text_path('SPLAT', 30, 0, 86, 0.12)[0] + '</g>'
    W = x0 + w1 + gap + w2 + 16
    body += f'<g fill="{sub}" transform="translate({x0+2} 0)">' + text_path('STEREO VIDEO TO GAUSSIAN SPLATS', 8.5, 0, 112, 0.32)[0] + '</g>'
    return svg(round(W), 140, body), round(W)
s, W = lockup('#F2F3F5', '#8A9099'); open('hs-logo-dark.svg','w').write(s)
s, W = lockup('#111317', '#5F6570', light=True); open('hs-logo-light.svg','w').write(s)
print('lockup width', W)

open('hs-mark-light.svg','w').write(svg(120,120,mark(light=True, id='ml')))
# small sizes (16-32 px): body + mark only, no lens detail
small = '''<defs><linearGradient id="sb" x1="0" y1="0" x2="0.35" y2="1"><stop offset="0" stop-color="#2A2D33"/><stop offset="1" stop-color="#0A0B0D"/></linearGradient></defs>
<rect x="100" y="100" width="824" height="824" rx="185" fill="url(#sb)" stroke="#4A4F57" stroke-width="10"/>
<g transform="translate(512 512) scale(6.2) translate(-60 -60)">''' + mark(id='sm') + '</g>'
open('hs-appicon-small.svg','w').write(svg(1024,1024,small))
