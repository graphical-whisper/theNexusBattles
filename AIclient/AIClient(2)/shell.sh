#!/bin/bash

# Variables
module="Product"     # Nombre de la clase (capitalizado)
bocchi="product"     # Nombre de la carpeta principal MVC

# Inicializar proyecto y dependencias
npm init -y
npm install express
npm install --save-dev typescript ts-node ts-node-dev @types/express @types/node
npm install -g typescript

# Crear tsconfig
npx tsc --init

# Estructura de carpetas
mkdir -p src/$bocchi/{controller,model,view,types,util,provider}
mkdir -p src/express scripts test database build assets env

# .gitignore
echo "/node_modules" > .gitignore

# Archivo .env
cat <<EOL > .env
PORT=1802
HOST=localhost
STATIC_DIR=../../assets/img
EOL

# src/index.ts
cat <<EOL > src/index.ts
import ${module}Controller from "./${bocchi}/controller/${module}Controller";
import Server from "./express/Server";
import ${module}View from "./${bocchi}/view/${module}View";
import ${module}Model from "./${bocchi}/model/${module}Model";
import ${module}Provider from "./${bocchi}/provider/${module}Provider";

const ${bocchi}Model = new ${module}Model();
const ${bocchi}Controller = new ${module}Controller(${bocchi}Model);
const ${bocchi}View = new ${module}View(${bocchi}Controller);
const env = new ${module}Provider();

const server = new Server(env, ${bocchi}View);
server.start();
EOL

# Util vacío
touch src/${bocchi}/util/JsonManager.ts
touch src/${bocchi}/types/${module}Interface.ts

# Controller
cat <<EOL > src/${bocchi}/controller/${module}Controller.ts
import { Request, Response } from "express";
import ${module}Model from "../model/${module}Model";
import ${module}Interface from "../types/${module}Interface";

export default class ${module}Controller {
    constructor(private readonly ${bocchi}Model: ${module}Model) {}
}
EOL

# Model
cat <<EOL > src/${bocchi}/model/${module}Model.ts
export default class ${module}Model {
    constructor() {}
}
EOL

# View
cat <<EOL > src/${bocchi}/view/${module}View.ts
import { Router } from "express";
import ${module}Controller from "../controller/${module}Controller";

export default class ${module}View {
    router: Router;

    constructor(private readonly ${bocchi}Controller: ${module}Controller) {
        this.router = Router();
        this.routes();
    }

    readonly routes = (): void => {
        // this.router.post(\`/${bocchi}s\`, this.${bocchi}Controller.create${module});
    }
}
EOL

# Provider
cat <<EOL > src/${bocchi}/provider/${module}Provider.ts
import Env from "../types/Env";

export default class ${module}Provider {
    private readonly env: Env;

    constructor() {
        this.env = { 
            PORT: Number(process.env["PORT"]) || 1818,
            HOST: process.env["HOST"] || "localhost",
            STATIC_DIR: process.env["STATIC_DIR"] || "../../assets/img"
        };
    }

    readonly HOST = () => this.env.HOST;
    readonly PORT = () => this.env.PORT;
    readonly STATIC_DIR = () => this.env.STATIC_DIR;
}
EOL

# Server
cat <<EOL > src/express/Server.ts
import express, { Application } from "express";
import ${module}View from "../${bocchi}/view/${module}View";
import path from "path";
import ${module}Provider from "../${bocchi}/provider/${module}Provider";

export default class Server {
    private readonly app: Application;

    constructor(
        private readonly env: ${module}Provider,
        private readonly ${bocchi}View: ${module}View
    ) {
        this.app = express();
        this.configure();
        this.static();
        this.routes();
    }

    readonly routes = (): void => {
        this.app.use("/", this.${bocchi}View.router);
    }

    readonly start = (): void => {
        const port = this.env.PORT();
        this.app.listen(port, () => {
            console.log(\`Server is running on http://\${this.env.HOST()}:\${port}\`);
        });
    }

    readonly static = (): void => {
        this.app.use(express.static(path.join(__dirname, this.env.STATIC_DIR())));
    }

    readonly configure = (): void => {
        this.app.use(express.json());
        this.app.use(express.urlencoded({ extended: true }));
    }
}
EOL

# env.ts
cat <<EOL > src/${bocchi}/types/env.ts
type Env ={
    PORT: number
    HOST: string
    STATIC_DIR: string
}

export default Env
EOL

# .env
cat <<EOL > env/.env
PORT=1802
HOST=localhost
STATIC_DIR=../../assets/img
EOL