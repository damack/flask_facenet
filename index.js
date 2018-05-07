var fs = require('fs');
var nj = require('numjs');
var express = require('express');
var multer = require('multer');

var app = express();
var {
    Facenet
} = require('facenet');
var upload = multer({ dest: 'uploads/' });

const facenet = new Facenet();
const distance = (source, destination) => {
    const broadCastedSource = nj.zeros(destination.shape);
    for (let i = 0; i < destination.shape[0]; i += 1) {
        broadCastedSource.slice([i, i + 1]).assign(source.reshape(1, -1), false);
    }

    const l2 = destination.subtract(broadCastedSource).pow(2).tolist();
    const distList = l2.map(numList => numList.reduce((prev, curr) => prev + curr, 0)).map(Math.sqrt);

    return distList;
};

app.get('/faces', (req, res) => {
    res.sendStatus(200);
});
app.post('/faces', upload.single('file'), async (req, res) => {
    const faces = await facenet.align(req.file.path);

    const data = [];
    for (const face of faces) {
        if (face.confidence > 0.99) {
            data.push({
                base64: face.dataUrl(),
                embedding: await facenet.embedding(face),
                location: face.location
            });
        }
    }

    fs.unlinkSync(req.file.path);
    res.send(data);
});

const main = async () => {
    await facenet.init();
    app.listen(5000, () => {
        console.log("App server started!");
    });
}
main();