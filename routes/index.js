const express = require('express');
const fs = require('fs');
const uuidv4 = require('uuid/v4');
const child_process  = require('child_process');
const multer  = require('multer')
const path = require("path");

const router = express.Router();


var results = [];

results[""] = "None";


var storage = multer.diskStorage({
  destination: function (req, file, cb) {

    const dirName =  "uploads/" + req.runDir;

    // make directory if it does not exist
    if (!fs.existsSync(dirName)){
      fs.mkdirSync(dirName, {recursive: true});
    }

    cb(null, dirName)
  },
  filename: function (req, file, cb) {

    cb(null, uuidv4());
  }
});

var upload = multer({ storage: storage });


/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});


router.get('/upload', function(req, res, next) {
  res.redirect('/');
});


router.post('/upload', function(req, res, next) {

  // generate the run directory name
  req.runDir =  new Date().getTime().toString();
  next();
}, upload.array('images'), function(req, res, next) {

  // put all the uploaded images into the /uploads directory
  // fork a process to run the python program
  // open a websocket so that it can let the client know whrn it is done
  // render the page that contains a link to where the user can check back to get the result image


  results[req.runDir] = null;


  const output = uuidv4();

  const run = child_process.spawn("./runImageProcessing.sh", [req.runDir, output]);

  run.on("close", (code) => {
    console.log(`stdout: done`);

    results[req.runDir] = {name: output, success: code === 0};

    req.io.emit(req.runDir, {done: true, message: "results are ready"});
  });


  res.render('upload', {
    host: req.protocol + "://" + req.headers.host,
    key: req.runDir
  });
});


router.get('/results', function(req, res, next) {

  const key = req.query.result || "";


  const result = results[key];

  if (key === "") {
    return res.render('result', { message: "You must specify the run you want to get the result for using ?result=result code" });
  }


  if (result === undefined) {
    return res.render('result', { message: "Sorry. No such run exists" });
  }


  if (result === null) {
    return res.render('upload', {
      host: req.protocol + "://" + req.headers.host,
      message: "Result is not ready yet, you will be redirected when the result is ready, or go reload the page later",
      key: key
    });
  }

  if (!result.success) {
    return res.render('upload', {
      host: req.protocol + "://" + req.headers.host,
      message: "Something went wrong, image can not be generated",
      key: key
    });
  }


  const rootDir = path.resolve(__dirname, "..");
  let filePath = path.join(rootDir, "generated", result.name);

  res.download("./public/generated/" + result.name, "generated.png");

});



module.exports = router;
