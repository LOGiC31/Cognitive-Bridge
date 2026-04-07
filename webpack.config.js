const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  entry: {
    'content/content': './extension/content/content.js',
    'background/service-worker': './extension/background/service-worker.js',
    'popup/popup': './extension/popup/popup.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js',
    clean: true,
  },
  resolve: {
    extensions: ['.js'],
    fallback: {
      fs: false,
      path: false,
      crypto: false,
    },
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: 'extension/manifest.json', to: 'manifest.json' },
        { from: 'extension/popup/popup.html', to: 'popup/popup.html' },
        { from: 'extension/popup/popup.css', to: 'popup/popup.css' },
        { from: 'extension/icons', to: 'icons' },
        { from: 'extension/data', to: 'data' },
        {
          from: 'node_modules/@xenova/transformers/dist/ort-wasm*.wasm',
          to: 'wasm/[name][ext]',
        },
      ],
    }),
  ],
  optimization: {
    minimize: true,
  },
  devtool: false,
};
