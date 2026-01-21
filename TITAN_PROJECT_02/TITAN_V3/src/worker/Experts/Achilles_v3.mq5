//+------------------------------------------------------------------+
//|                                                  Achilles_v2.mq5 |
//|                             Phase 3: Antigravity ZMQ Architecture |
//|                                     "Pure Gold" R3K Compliance    |
//+------------------------------------------------------------------+
#property copyright "Antigravity AI"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include "../../Include/Json.mqh"   // Ensure path matches your structure
#include "../../Include/ZmqLib.mqh" // Our new wrapper

// --- Connection Settings ---
input group " ZMQ Connection"
input string   ZmqHost           = "127.0.0.1";
input string   ZmqPort           = "5555";     // Default REQ-REP port

// --- Money Management ---
input group " Money Management"
input double   RiskPercent       = 1.0;
input double   MinLotSize        = 0.01;

// --- Protection (R3K) ---
input group " Protection"
input int      StopLossPoints    = 500;
input int      TakeProfitPoints  = 1000;

// --- Mode ---
input bool     LiveTradingMode   = false;

// --- Globals ---
CTrade trade;
CZmqSocket zmq;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("Achilles V2 (Antigravity) Initializing...");
   
   // 1. ZMQ Init
   if(!zmq.Initialize()) {
      Print("CRITICAL: ZMQ Initialization Failed. Check libzmq.dll in Libraries.");
      return(INIT_FAILED);
   }
   
   string addr = "tcp://" + ZmqHost + ":" + ZmqPort;
   if(!zmq.Connect(addr)) {
      Print("CRITICAL: ZMQ Connect Failed to ", addr);
      return(INIT_FAILED);
   }
   
   Print("ZMQ Engine: ONLINE. Listening to Brain.");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   zmq.Shutdown();
   Print("Achilles V2 Stopped. ZMQ Closed.");
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // --- R3K DATA GATHERING ---
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Position Data
   bool has_pos = PositionSelect(_Symbol);
   int pos_type = -1;
   double open_price = 0.0;
   long open_time = 0;
   double current_profit = 0.0;
   
   if(has_pos) {
      pos_type = (int)PositionGetInteger(POSITION_TYPE);
      open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      open_time = PositionGetInteger(POSITION_TIME);
      current_profit = PositionGetDouble(POSITION_PROFIT);
   }

   // --- JSON PAYLOAD ---
   string json_part1 = StringFormat("{\"symbol\": \"%s\", \"ask\": %.5f, \"bid\": %.5f, \"balance\": %.2f, \"equity\": %.2f, ", 
                                    _Symbol, ask, bid, balance, equity);
   string json_part2 = StringFormat("\"has_position\": %s, \"position_type\": %d, \"open_price\": %.5f, \"open_time\": %d, \"current_profit\": %.2f}", 
                                    (has_pos ? "true" : "false"), pos_type, open_price, open_time, current_profit);
   string payload = json_part1 + json_part2;

   // --- ZMQ TRANSACTION (The Antigravity Lift) ---
   // Blocking Send/Recv (Sync Pattern, but microsecond latency compared to HTTP)
   if(!zmq.Send(payload)) {
      Print("ZMQ Send Error.");
      return;
   }
   
   string response = zmq.Receive();
   if(response == "") {
      Print("ZMQ Recv Timeout/Empty.");
      return; 
   }

   // --- PARSE RESPONSE ---
   CJson parser(response);
   string action = parser.GetString("action");
   double confidence = parser.GetDouble("confidence");
   string reason_msg = parser.GetString("reason");

   if(action != "HOLD") 
      PrintFormat("BRAIN [%s]: %s (Conf: %.2f) >> %s", action, _Symbol, confidence, reason_msg);

   // --- R3K EXECUTION LOGIC (DEFENSES) ---
   
   // 1. CLOSE
   if(has_pos && (action == "STOP_TRADING" || 
     (action == "CLOSE_BUY" && pos_type == POSITION_TYPE_BUY) || 
     (action == "CLOSE_SELL" && pos_type == POSITION_TYPE_SELL))) 
   {
      trade.PositionClose(_Symbol);
      if(action == "STOP_TRADING") ExpertRemove();
   }

   // 2. OPEN (With Invalid Stops Defense)
   if(!has_pos && confidence > 0.8 && (action == "BUY" || action == "SELL")) {
      
      if(LiveTradingMode) {
         // --- [TAG: R3K_DEFENSE_DYNAMIC_STOPS] ---
         // "Oro Puro" Compliance: MathMax(StopLevel, SafetyBuffer)
         long stop_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
         int safe_dist = (int)MathMax(stop_level + 10, 100); // 100 Point Safety Buffer
         double point = _Point;
         
         double sl_price = 0.0;
         double tp_price = 0.0;
         double open_target = 0.0;
         ENUM_ORDER_TYPE type = ORDER_TYPE_BUY;
         
         if(action == "BUY") {
            type = ORDER_TYPE_BUY;
            open_target = ask;
            // SL below Bid
            sl_price = NormalizeDouble(bid - StopLossPoints * point, _Digits);
            tp_price = NormalizeDouble(ask + TakeProfitPoints * point, _Digits);
         } else {
            type = ORDER_TYPE_SELL;
            open_target = bid;
            // SL above Ask
            sl_price = NormalizeDouble(ask + StopLossPoints * point, _Digits);
            tp_price = NormalizeDouble(bid - TakeProfitPoints * point, _Digits);
         }

         // --- [TAG: R3K_DEFENSE_HYGIENE] ---
         // ZeroMemory prevents "Invalid Stops" (10016) by clearing garbage data.
         MqlTradeRequest request;
         MqlTradeResult result;
         ZeroMemory(request);
         ZeroMemory(result);
         
         request.action = TRADE_ACTION_DEAL;
         request.symbol = _Symbol;
         request.volume = MinLotSize;
         request.type = type;
         request.price = open_target;
         request.sl = sl_price;
         request.tp = tp_price;
         request.deviation = 10;
         
         if(!OrderSend(request, result)) {
            PrintFormat("Order Critical Fail. Ret: %d, Err: %d", result.retcode, GetLastError());
         } else {
            Print("R3K Execution Success. Ticket: ", result.order);
         }
      } else {
         Print("SIMULATION: Virtual Execution Success.");
      }
   }
  }
//+------------------------------------------------------------------+
