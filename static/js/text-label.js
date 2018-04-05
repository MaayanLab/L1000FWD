function makeTextCanvas( message, x, y, z, parameters ){
	if ( parameters === undefined ) parameters = {}; 
    var fontface = parameters.hasOwnProperty("fontface") ?  
        parameters["fontface"] : "Arial";      
    var fontsize = parameters.hasOwnProperty("fontsize") ?  
        parameters["fontsize"] : 18; 
    var textColor = parameters.hasOwnProperty("textColor") ? 
        parameters["textColor"] : { r:0, g:0, b:255, a:1.0 }; 

    var canvas = document.createElement('canvas'); 
    var context = canvas.getContext('2d'); 

    canvas.width = WIDTH; 
    canvas.height = HEIGHT; 

    context.font = fontsize + "px " + fontface; 
    context.textBaseline = "alphabetic"; 

    context.textAlign = "left"; 
    // get size data (height depends only on font size) 
    var metrics = context.measureText( message ); 
    var textWidth = metrics.width; 

    // text color.  Note that we have to do this AFTER the round-rect as it also uses the "fillstyle" of the canvas 
    context.fillStyle = getCanvasColor(textColor); 

    // calculate the project of 3d point into 2d plain
    var point = new THREE.Vector3(x, y, z);
    var pv = new THREE.Vector3().copy(point).project(camera);
    var coords = {
    	x: ((pv.x + 1) / 2 * WIDTH) * DPR, 
    	y: -((pv.y - 1) / 2 * HEIGHT) * DPR
    };
    // draw the text
    context.fillText(message, coords.x, coords.y)

    // styles of canvas element
    canvas.style.left = 0;
    canvas.style.top = 0;
    canvas.style.position = 'absolute';
    canvas.style.pointerEvents = 'none';

    return canvas;
}

/** 
 * convenience for converting JSON color to rgba that canvas wants
 * Be nice to handle different forms (e.g. no alpha, CSS style, etc.)
 */ 
function getCanvasColor ( color ) { 
    return "rgba(" + color.r + "," + color.g + "," + color.b + "," + color.a + ")"; 
} 