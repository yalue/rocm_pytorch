# Pytorch Profiler

### Install & Build

1. install [Node.js](https://nodejs.org/)
  * ```bash
    curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    sudo apt-get install -y nodejs```
2. install [Yarn](https://yarnpkg.com/)
  * ```bash
     curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
     echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
     sudo apt update && sudo apt install yarn
  ```
3. shell `yarn`
4. shell `yarn build`
5. `./dist/index.html`