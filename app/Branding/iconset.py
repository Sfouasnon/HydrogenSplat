import asyncio, os
from playwright.async_api import async_playwright
SIZES = [('icon_16x16',16,'small'),('icon_16x16@2x',32,'small'),('icon_32x32',32,'small'),('icon_32x32@2x',64,'full'),
         ('icon_128x128',128,'full'),('icon_128x128@2x',256,'full'),('icon_256x256',256,'full'),('icon_256x256@2x',512,'full'),
         ('icon_512x512',512,'full'),('icon_512x512@2x',1024,'full')]
async def main():
    os.makedirs('AppIcon.iconset', exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for name, px, kind in SIZES:
            pg = await b.new_page(viewport={'width':px,'height':px})
            src = os.path.abspath('hs-appicon-small.svg' if kind=='small' else 'hs-appicon.svg')
            open('_one.html','w').write(f'<body style="margin:0;background:transparent"><img src="file://{src}" width="{px}" height="{px}" style="display:block"></body>')
            await pg.goto('file://'+os.path.abspath('_one.html'))
            await pg.wait_for_timeout(100)
            await pg.screenshot(path=f'AppIcon.iconset/{name}.png', omit_background=True)
            await pg.close()
        pg = await b.new_page(viewport={'width':1300,'height':260})
        open('_prev.html','w').write(f'''<body style="margin:0;display:flex;gap:24px;padding:20px;align-items:end;background:linear-gradient(90deg,#1a1b1f 50%,#f3f3f1 50%)">
        {''.join(f'<img src="file://{os.path.abspath("AppIcon.iconset/"+n+".png")}" width="{min(px,128)}">' for n,px,k in SIZES if '@2x' not in n)}
        <img src="file://{os.path.abspath('hs-mark-light.svg')}" width="120"><img src="file://{os.path.abspath('hs-logo-light.svg')}" width="420"></body>''')
        await pg.goto('file://'+os.path.abspath('_prev.html'))
        await pg.wait_for_timeout(200)
        await pg.screenshot(path='preview2.png')
        await b.close()
asyncio.run(main())
