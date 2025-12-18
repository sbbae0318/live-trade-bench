import React from 'react';
import BinanceChart from './components/BinanceChart';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Binance Trading Chart</h1>
        <p>실시간 Binance 거래 데이터 시각화</p>
      </header>
      <main className="App-main">
        <BinanceChart />
      </main>
    </div>
  );
}

export default App;



