FROM node:10

COPY package*.json ./

RUN npm install
COPY . .

EXPOSE 3000

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 --no-cache-dir install opencv-python
RUN pip3 --no-cache-dir install numpy
RUN pip3 --no-cache-dir install scipy
RUN pip3 --no-cache-dir install matplotlib



CMD [ "node", "app.js" ]
