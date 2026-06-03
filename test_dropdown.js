const puppeteer = require('puppeteer');   

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    // Catch console errors
    page.on('console', msg => console.log('BROWSER CONSOLE:', msg.type(), msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
    
    await page.goto('http://127.0.0.1:8000', {waitUntil: 'networkidle0'});
    
    // Check if district dropdown exists
    const trigger = await page.$('#districtTrigger');
    if (!trigger) {
        console.log('🔴 Dropdown trigger not found in DOM!');
    } else {
        console.log('🟢 Dropdown trigger found.');
        
        // Click it
        await trigger.click();
        console.log('✅ Clicked trigger.');
        
        // Wait a bit
        await new Promise(r => setTimeout(r, 500));
        
        // Check if panel is visible
        const showClass = await page.evaluate(() => document.querySelector('#districtPanel').classList.contains('show'));
        console.log('Panel has "show" class?', showClass);
        
        // Check options
        const opts = await page.evaluate(() => document.querySelectorAll('#districtList li').length);
        console.log('Number of district options:', opts);
    }
    
    await browser.close();
})();
