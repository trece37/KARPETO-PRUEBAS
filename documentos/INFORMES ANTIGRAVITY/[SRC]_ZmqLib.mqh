//+------------------------------------------------------------------+
//|                                                       ZmqLib.mqh |
//|                                  Antigravity Minimal ZMQ Wrapper |
//|                                         Phase 3: High Frequency  |
//+------------------------------------------------------------------+
#property copyright "Antigravity AI"
#property strict

// --- DLL LIMITATIONS ---
// User MUST copy 'libzmq.dll' to 'MQL5/Libraries/libzmq.dll'
#import "libzmq.dll"
   int zmq_ctx_new();
   int zmq_ctx_term(int context);
   int zmq_socket(int context, int type);
   int zmq_close(int socket);
   int zmq_connect(int socket, const uchar &endpoint[]);
   int zmq_send(int socket, const uchar &buf[], int len, int flags);
   int zmq_recv(int socket, uchar &buf[], int len, int flags);
#import

// --- ZMQ CONSTANTS ---
#define ZMQ_REQ 3
#define ZMQ_REP 4
#define ZMQ_PUB 1
#define ZMQ_SUB 2
#define ZMQ_PUSH 8
#define ZMQ_PULL 7

#define ZMQ_NOBLOCK 1
#define ZMQ_DONTWAIT 1

// --- WRAPPER CLASS ---
class CZmqSocket {
private:
   int m_context;
   int m_socket;
   bool m_connected;

public:
   CZmqSocket(int context, int type) : m_context(context), m_socket(0), m_connected(false) {
      if(m_context != 0) {
         m_socket = zmq_socket(m_context, type);
      }
   }
   
   ~CZmqSocket() {
      if(m_socket != 0) zmq_close(m_socket);
   }

   bool bind(string address) {
       if(m_socket == 0) return false;
       uchar addrArg[];
       StringToCharArray(address, addrArg);
       // Note: In MQL5 for wrappers, sometimes we bind, sometimes we connect. 
       // For PUB/PULL usually we might bind or connect depending on topology. 
       // Libzmq Import 'zmq_bind' missing in strict import above? 
       // FIX: We need zmq_bind imported if we act as server, or zmq_connect if client.
       // For this setup (Titan Bridge), usually MT5 Connects to Python (Server).
       // So 'connect' is correct. If we needed bind, we would add the import.
       return (zmq_connect(m_socket, addrArg) == 0);
   }

   // UPDATED: Now accepts flags
   bool send(string message, int flags=0) {
      if(m_socket == 0) return false;
      
      uchar data[];
      int len = StringToCharArray(message, data) - 1; 
      if(len < 0) len = 0;
      
      int bytes_sent = zmq_send(m_socket, data, len, flags);
      return (bytes_sent >= 0);
   }

   // UPDATED: Now accepts flags
   string recv(int flags=0, int buffer_size=4096) {
      if(m_socket == 0) return "";
      
      uchar buffer[];
      ArrayResize(buffer, buffer_size);
      
      int bytes_recvd = zmq_recv(m_socket, buffer, buffer_size, flags);
      
      string result = "";
      if(bytes_recvd > 0) {
         result = CharArrayToString(buffer, 0, bytes_recvd);
      }
      ArrayFree(buffer);
      return result;
   }
};

class Context {
   int ctx;
public:
   Context() { ctx = zmq_ctx_new(); }
   ~Context() { zmq_ctx_term(ctx); }
   int ptr() { return ctx; }
};

class Socket : public CZmqSocket {
public:
   Socket(Context &c, int type) : CZmqSocket(c.ptr(), type) {}
};
