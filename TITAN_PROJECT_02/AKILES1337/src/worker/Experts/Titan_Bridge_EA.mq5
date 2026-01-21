//+------------------------------------------------------------------+
//|                                              Titan_Bridge_EA.mq5 |
//|                                   Copyright 2026, Antigravity AI |
//|                                             https://antigravity.ai|
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity AI"
#property link      "https://antigravity.ai"
#property version   "1.00"
#property strict

#include <Zmq/ZmqBridge.mqh>

//--- input parameters
input string   ZmqHost     = "127.0.0.1";
input int      PubPort     = 5556;
input int      PullPort    = 5557;

CZmqBridge *bridge;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   bridge = new CZmqBridge(ZmqHost, PubPort, PullPort);
   
   if(!bridge.Connect())
   {
      Print("❌ ZMQ BRIDGE: Connection Failed!");
      return(INIT_FAILED);
   }
   
   Print("🚀 TITAN BRIDGE EA: Ready and Listening.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   delete bridge;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // 1. Enviar Datos al Cerebro
   bridge.SendTick(bid, ask, balance, equity);
   
   // 2. Poll por Órdenes
   string orderMsg = bridge.PollOrder();
   if(orderMsg != "")
   {
      Print("📩 ORDEN RECIBIDA DE PYTHON: ", orderMsg);
      // Aquí iría la ejecución real de OrderSend()
   }
}
//+------------------------------------------------------------------+
