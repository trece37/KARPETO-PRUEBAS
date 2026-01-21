//+------------------------------------------------------------------+
//|                                                  Achilles_v1.mq5 |
//|                                  Copyright 2024, Manel & David. |
//|                                             https://www.google.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Manel & David."
#property link      "https://www.google.com"
#property version   "1.00"
#property strict

// --- Connection Settings ---
input group " Brain Connection"
input string   BrainURL          = "http://127.0.0.1:8000"; // Brain API URL

// --- Money Management ---
input group " Money Management"
input double   RiskPercent       = 1.0;      // Risk per trade (% of Balance)
input double   MaxLotSize        = 10.0;     // Maximum allowed Lot Size
input double   MinLotSize        = 0.01;     // Minimum allowed Lot Size

// --- Trade Settings ---
input group " Trade Settings"
input int      MagicNumber       = 1337;     // Magic Number (ID)
input int      Slippage          = 3;        // Max Slippage (points)
input int      MaxSpread         = 20;       // Max Spread allowed (points)

// --- Protection ---
input group " Protection"
input int      StopLossPoints    = 500;      // Stop Loss (Points)
input int      TakeProfitPoints  = 1000;     // Take Profit (Points)
input bool     UseTrailingStop   = true;     // Enable Trailing Stop
input int      TrailingStopPoints= 200;      // Trailing Stop Distance (Points)
input int      TrailingStep      = 50;       // Trailing Step (Points)

// --- Time Filters ---
input group " Time Filters"
input bool     TradeOnFriday     = true;     // Allow Trading on Friday?
// --- Execution Mode ---
input group " Execution Mode"
input bool     LiveTradingMode   = false;    // TRUE = Real Trades, FALSE = Simulation (Logs only)

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Allow WebRequest
   // Note: User must add URL to Tools -> Options -> Expert Advisors -> Allow WebRequest
   
   Print("Achilles Bot Initialized.");
   Print("Mode: ", LiveTradingMode ? "LIVE TRADING (CAUTION)" : "SIMULATION");
   Print("Connecting to Brain at: ", BrainURL);
   
   return(INIT_SUCCEEDED);
  }
// ... (Initial part remains, jumping to OnTick execution) ...

      // OPEN Logic
      if(!has_pos && action == "BUY" && confidence > 0.8)
        {
         if(LiveTradingMode)
           {
            double sl = bid - StopLossPoints * _Point;
            double tp = bid + TakeProfitPoints * _Point;
            if(trade.Buy(MinLotSize, _Symbol, ask, sl, tp, "Achilles AI Buy"))
               Print("BUY Order Executed. Ticket: ", trade.ResultOrder());
            else
               Print("BUY Execution Failed. Error: ", GetLastError());
           }
         else
           {
            Print("SIMULATION: Brain wants to BUY! (Confidence: ", confidence, ")");
           }
        }
      
      if(!has_pos && action == "SELL" && confidence > 0.8)
        {
         if(LiveTradingMode)
           {
            double sl = ask + StopLossPoints * _Point;
            double tp = ask - TakeProfitPoints * _Point;
            if(trade.Sell(MinLotSize, _Symbol, bid, sl, tp, "Achilles AI Sell"))
               Print("SELL Order Executed. Ticket: ", trade.ResultOrder());
            else
               Print("SELL Execution Failed. Error: ", GetLastError());
           }
         else
           {
             Print("SIMULATION: Brain wants to SELL! (Confidence: ", confidence, ")");
           }
        }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Print("Achilles Bot Stopped.");
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include "../Include/Json.mqh"

CTrade trade; // Execution Object

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // 1. Gather Market & Account Data
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // 2. Gather Position Data
   bool has_pos = PositionSelect(_Symbol);
   int pos_type = -1;
   double open_price = 0.0;
   long open_time = 0;
   double current_profit = 0.0;
   
   if(has_pos)
     {
      pos_type = (int)PositionGetInteger(POSITION_TYPE); // 0=Buy, 1=Sell
      open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      open_time = PositionGetInteger(POSITION_TIME);
      current_profit = PositionGetDouble(POSITION_PROFIT);
     }
   
   // 3. Construct JSON Payload
   // Note: StringFormat with many args can be tricky in MQL5, splitting for safety
   string json_part1 = StringFormat("{\"symbol\": \"%s\", \"ask\": %.5f, \"bid\": %.5f, \"balance\": %.2f, \"equity\": %.2f, ", 
                                    _Symbol, ask, bid, balance, equity);
                                    
   string json_part2 = StringFormat("\"has_position\": %s, \"position_type\": %d, \"open_price\": %.5f, \"open_time\": %d, \"current_profit\": %.2f}", 
                                    (has_pos ? "true" : "false"), pos_type, open_price, open_time, current_profit);
                                    
   string payload = json_part1 + json_part2;
   
   // 4. Send to Brain
   char post_data[];
   StringToCharArray(payload, post_data);
   char result_data[];
   string result_headers;
   string url = BrainURL + "/predict";
   
   ResetLastError();
   int timeout = 5000;
   int res = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, post_data, result_data, result_headers);
   
   // 5. Process Brain Response
   if(res == 200)
     {
      string json_response = CharArrayToString(result_data);
      CJson parser(json_response);
      
      string action = parser.GetString("action");
      double confidence = parser.GetDouble("confidence");
      string reason = parser.GetString("reason");
      
      if(action != "HOLD") 
         PrintFormat("BRAIN SIGNAL: %s | Conf: %.2f | Reason: %s", action, confidence, reason);
      
      // --- Execution Logic ---
      
      // CLOSE Logic
      if((action == "CLOSE_BUY" && pos_type == POSITION_TYPE_BUY) || 
         (action == "CLOSE_SELL" && pos_type == POSITION_TYPE_SELL) ||
         (action == "STOP_TRADING" && has_pos))
        {
         trade.PositionClose(_Symbol);
         Print("Closing Position by Brain Command.");
        }
        
      // OPEN Logic
      if(!has_pos && action == "BUY" && confidence > 0.8)
        {
         if(LiveTradingMode)
           {
            // Simple Lot Calc (Fixed for now)
            double sl = bid - StopLossPoints * _Point;
            double tp = bid + TakeProfitPoints * _Point;
            
            if(trade.Buy(MinLotSize, _Symbol, ask, sl, tp, "Achilles AI Buy"))
               Print("BUY Order Executed. Ticket: ", trade.ResultOrder());
            else
               Print("BUY Execution Failed. Error: ", GetLastError());
           }
         else
           {
            Print("SIMULATION: Brain wants to BUY! (Conf: ", confidence, ")");
           }
        }
      
      if(!has_pos && action == "SELL" && confidence > 0.8)
        {
         if(LiveTradingMode)
           {
            double sl = ask + StopLossPoints * _Point;
            double tp = ask - TakeProfitPoints * _Point;
            
            if(trade.Sell(MinLotSize, _Symbol, bid, sl, tp, "Achilles AI Sell"))
               Print("SELL Order Executed. Ticket: ", trade.ResultOrder());
            else
               Print("SELL Execution Failed. Error: ", GetLastError());
           }
         else
           {
            Print("SIMULATION: Brain wants to SELL! (Conf: ", confidence, ")");
           }
        }
        
      // EMERGENCY STOP
      if(action == "STOP_TRADING")
        {
         Print("CRITICAL: Brain triggered Circuit Breaker! Stopping EA.");
         ExpertRemove(); // Unload EA
        }
     }
   else
     {
      // Connection Error Handling
      // Print("Error contacting Brain: ", GetLastError());
     }
  }
//+------------------------------------------------------------------+
