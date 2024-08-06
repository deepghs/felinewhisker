function addHotKeyListeners() {
    document.addEventListener('keydown', function (event) {
        if (event.key === 'ArrowLeft') {  // left
            document.querySelector('#left-button').click();
        } else if (event.key === 'ArrowRight') { // right
            document.querySelector('#right-button').click();
        } else if (event.ctrlKey && event.key === 's') {  // ctrl+s
            event.preventDefault();
            document.querySelector('#save-button').click();
        }
    });

    let timeout;

    function resetAutoSaveTimer() {
        clearTimeout(timeout);
        timeout = setTimeout(function () {
            console.log('Auto save disabled.')
            // console.log('Auto save enabled.')
            // document.querySelector('#save-button').click();
        }, 30000);
    }

    document.addEventListener('mousemove', resetAutoSaveTimer, false);
    document.addEventListener('keypress', resetAutoSaveTimer, false);
    document.addEventListener('click', resetAutoSaveTimer, false);

    resetAutoSaveTimer();
}