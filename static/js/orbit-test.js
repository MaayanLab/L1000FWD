// modified from https://github.com/mrdoob/three.js/blob/dev/examples/misc_controls_orbit.html
if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

var camera, controls, scene, renderer, raycaster, points;
var projector;
var mouse = new THREE.Vector2();
var pointCount = 10000;
var colors = new Float32Array( pointCount * 3 );
var container;
var WIDTH = window.innerWidth;
var HEIGHT = window.innerHeight;
var DPR = window.devicePixelRatio;

init();
render(); // remove when using next line for animation loop (requestAnimationFrame)

// animate();
function init() {
    var WIDTH = window.innerWidth;
    var HEIGHT = window.innerHeight;
    var aspectRatio = WIDTH / HEIGHT;

    projector = new THREE.Projector();
    // The scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2( 0xcccccc, 0.002 );
    // The renderer
    renderer = new THREE.WebGLRenderer();
    renderer.setClearColor( scene.fog.color );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( WIDTH, HEIGHT );
    // Put the renderer's DOM into the container
    container = document.body;
    renderer.domElement.id = "renderer";
    container.appendChild( renderer.domElement );
    // The camera
    camera = new THREE.PerspectiveCamera( 45, aspectRatio, 1, 1000 );
    camera.position.z = 300;
    // The controls for zooming and panning behaviors
    controls = new THREE.OrbitControls( camera, renderer.domElement );
    controls.addEventListener( 'change', render ); // remove when using animation loop
    // enable animation loop when using damping or autorotation
    // controls.enableDamping = true;
    // controls.dampingFactor = 0.25;
    controls.enableZoom = true;

    // scatter plot
    // var scatterPlot = new THREE.Object3D();
    // scene.add(scatterPlot)
    // scatterPlot.rotation.y = 0

    // var mat = new THREE.PointsMaterial(
    //   {
    //     vertexColors: THREE.VertexColors,
    //     size: 1.5,
    //     opacity: 0.6,
    //     transparent: true,
    // });

    // var pointCount = 10000;
    // arrays to construct BufferGeometry
    var indices = new Uint32Array( pointCount );
    var positions = new Float32Array( pointCount * 3 );
    // var colors = new Float32Array( pointCount * 3 );
    var sizes = new Float32Array( pointCount );
    var labels = new Array( pointCount );

    var color = new THREE.Color()

    for (var i=0; i<pointCount; i++) {
        indices[i] = i;

      var x = (Math.random()-0.5) * 100 
      var y = (Math.random()-0.5) * 100
      var z = (Math.random()-0.5) * 100

      positions[ i*3 ] = x;
      positions[ i*3 + 1 ] = y; 
      positions[ i*3 + 2 ] = z; 

      // color.setRGB( (x+50)/100, (y+50)/100, (z+50)/100 )

      color.setRGB( 0.8, 0.1, 0.1 )
      color.toArray( colors, i * 3 );

      sizes[ i ] = 5.0
      labels[ i ] = 'p' + i;
      // pointGeo.vertices.push(new THREE.Vector3(x,y,z));

      // pointGeo.colors.push(new THREE.Color()
      //   .setRGB(
      //   (x+50)/100, (z+50)/100, (y+50)/100));
      // var mesh = new THREE.Mesh(pointGeo, mat);
      // mesh.position.x = Math.random() * 100 
      // mesh.position.y = Math.random() * 100 
      // mesh.position.z = Math.random() * 100 
      // mesh.updateMatrix();
      // mesh.matrixAutoUpdate = false;
      // scene.add(mesh)
    }
    // var points = new THREE.Points(pointGeo, mat);
    // scatterPlot.add(points);
    // colors = new THREE.BufferAttribute( colors, 3 )

    var geometry = new THREE.BufferGeometry();
    geometry.setIndex( new THREE.BufferAttribute( indices, 1 ) );
    geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
    // geometry.addAttribute( 'normal', new THREE.BufferAttribute( normals, 3, true ) );
    geometry.addAttribute( 'size', new THREE.BufferAttribute( sizes, 1 ) );
    geometry.addAttribute( 'color', new THREE.BufferAttribute( colors.slice(), 3 ) );
    geometry.addAttribute( 'label', new THREE.BufferAttribute( labels, 1 ) );

    geometry.computeBoundingSphere();

    // var texture = new THREE.TextureLoader().load( "lib/textures/sprites/ball.png" );
    // texture.wrapS = THREE.RepeatWrapping;
    // texture.wrapT = THREE.RepeatWrapping;
    // console.log(texture)

    // var uniforms ={
    //                 amplitude: { value: 1.0 },
    //                 color:     { value: new THREE.Color( 0xffffff ) },
    //                 texture:   { value: texture }
    //             };


    var material = new THREE.PointsMaterial(
      {
        vertexColors: THREE.VertexColors,
        size: 2,
        opacity: 0.6,
        transparent: true,
    });


    // var canvas2 = document.createElement( 'canvas' );
    // canvas2.width = 128;
    // canvas2.height = 128;
    // var context = canvas2.getContext( '2d' );
    // context.arc( 64, 64, 64, 0, 2 * Math.PI );
    // context.fillStyle = 'rgb(0,127,255)';
    // context.fill();
    // var texture = new THREE.Texture( canvas2 );
    // texture.needsUpdate = true;

    // var material = new THREE.PointsMaterial({ 
    //     size: 2.0, 
    //     // map: texture, 
    //     transparent: true, 
    //     alphaTest: .1 
    // });



    points = new THREE.Points( geometry, material );
    scene.add( points );


    // lights
    // light = new THREE.DirectionalLight( 0xffffff );
    // light.position.set( 1, 1, 1 );
    // scene.add( light );
    // light = new THREE.DirectionalLight( 0x002288 );
    // light.position.set( -1, -1, -1 );
    // scene.add( light );
    // light = new THREE.AmbientLight( 0x222222 );
    // scene.add( light );
    //
    //

    // Mouse events
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.5;
    
    // window.addEventListener( 'click', onMouseMove, false );

    function onMouseMove( event ) {

        // calculate mouse position in normalized device coordinates
        // (-1 to +1) for both components
        event.preventDefault();

        mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
        mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

        render();
    }


    window.addEventListener( 'resize', onWindowResize, false );
    document.addEventListener( 'mousemove', onMouseMove, false );

}


function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
}
function animate() {
    requestAnimationFrame( animate );
    controls.update(); // required if controls.enableDamping = true, or if controls.autoRotate = true
    render();
}
function render() {

    // update the picking ray with the camera and mouse position
    raycaster.setFromCamera( mouse, camera );

    // calculate objects intersecting the picking ray
    var intersects = raycaster.intersectObject( points );

    points.geometry.attributes.color.needsUpdate = true;

    // reset colors
    points.geometry.attributes.color.array = colors.slice();
    points.geometry.computeBoundingSphere();
    points.updateMatrix();
    
    // remove text-label if exists
    var textLabel = document.getElementById('text-label')
    if (textLabel){
        textLabel.remove();
    }

    // add interactivities if there is intesecting points
    if ( intersects.length > 0 ) {
        // console.log(intersects)
        // only highlight the closest object
        var intersect = intersects[0];
        console.log(intersect)
        var idx = intersect.index;
        // change color of the point
        points.geometry.attributes.color.array[idx*3] = 0.1;
        points.geometry.attributes.color.array[idx*3+1] = 0.8;
        points.geometry.attributes.color.array[idx*3+2] = 0.1;
        // add text canvas

        // find the position of the point
        var pointPosition = { 
            x: points.geometry.attributes.position.array[idx*3],
            y: points.geometry.attributes.position.array[idx*3+1],
            z: points.geometry.attributes.position.array[idx*3+2],
        }


        var textCanvas = makeTextCanvas( points.geometry.attributes.label.array[idx], 
            pointPosition.x, pointPosition.y, pointPosition.z,
            { fontsize: 24, fontface: "Ariel", textColor: {r:0, g:0, b:255, a:1.0} }); 

        
        textCanvas.id = "text-label"
        document.body.appendChild(textCanvas);

       // points.geometry.computeBoundingSphere();
    }

    renderer.render( scene, camera );
}
