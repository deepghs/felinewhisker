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
}