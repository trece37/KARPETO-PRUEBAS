//+------------------------------------------------------------------+
//|                                                    ZmqBridge.mqh |
//|                                   Copyright 2026, Antigravity AI |
//|                                             https://antigravity.ai|
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity AI"
#property link      "https://antigravity.ai"
#property strict

#include <ZmqLib.mqh>
#include <Json.mqh>

class CZmqBridge
{
private:
   Context        m_context;
   Socket         *m_pub_socket;  // Envia ticks
   Socket         *m_pull_socket; // Recibe órdenes
   
   string         m_host;
   int            m_pub_port;
   int            m_pull_port;

public:
   CZmqBridge(string host, int pub_port, int pull_port)
   {
      m_host = host;
      m_pub_port = pub_port;
      m_pull_port = pull_port;
      
      // Inicializar Sockets usando el Wrapper Wrapper
      // PUB sends updates to subscribers
      m_pub_socket = new Socket(m_context, ZMQ_PUB);
      // PULL pulls commands from pushers
      m_pull_socket = new Socket(m_context, ZMQ_PULL);
   }

   ~CZmqBridge()
   {
      if(CheckPointer(m_pub_socket) == POINTER_DYNAMIC) delete m_pub_socket;
      if(CheckPointer(m_pull_socket) == POINTER_DYNAMIC) delete m_pull_socket;
   }

   bool Connect()
   {
      // En nuestra topología, MT5 suele hacer BIND a los puertos
      // Porque Python se conecta a MT5.
      // Python (Connect tcp://127.0.0.1:5556) -> MT5 (Bind *:5556)
      if(!m_pub_socket.bind(StringFormat("tcp://*:%d", m_pub_port))) {
         Print("❌ ZMQ PUB Bind Failed on Port ", m_pub_port);
         return false;
      }
      if(!m_pull_socket.bind(StringFormat("tcp://*:%d", m_pull_port))) {
         Print("❌ ZMQ PULL Bind Failed on Port ", m_pull_port);
         return false;
      }
      
      Print(StringFormat("✅ ZMQ BRIDGE: Bound to Ports %d (PUB) and %d (PULL)", m_pub_port, m_pull_port));
      return true;
   }

   void SendTick(double bid, double ask, double balance, double equity)
   {
      CJsonObject json;
      json.Set("bid", bid);
      json.Set("ask", ask);
      json.Set("balance", balance);
      json.Set("equity", equity);
      json.Set("timestamp", (int)TimeCurrent());

      // PRO FIX: Usar bandera ZMQ_NOBLOCK (Don't Wait) explícitamente
      // Esto asegura que si nadie escucha, MT5 no se congela ni un milisegundo.
      string msg = json.Serialize();
      m_pub_socket.send(msg, ZMQ_NOBLOCK); 
   }

   string PollOrder()
   {
      // PRO FIX: Usar ZMQ_NOBLOCK para chequear inbox
      // Retorna "" inmediatamente si no hay nada.
      string msg = m_pull_socket.recv(ZMQ_NOBLOCK);
      return msg; // Será string vacío si no hay datos
   }
};
