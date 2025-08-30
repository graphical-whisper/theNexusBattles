import * as readline from 'readline';

export class Readline {
    constructor(){}

    public pregunta(pregunta: string): Promise<string> {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        return new Promise((resolve) => {
            rl.question(pregunta, (respuesta) => {
                resolve(respuesta);
                rl.close();
            });
        });
    }
}