# integrated_model12.py
# Compact upgrade: model9_1 auth + model10 features + Alpaca paper trading
import time, datetime, logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import requests
import streamlit as st
import integrated_model10 as m10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Config ----
GEMINI_API_KEY = st.secrets.get('GEMINI_API_KEY', '')
ALPHA_VANTAGE_API_KEY = st.secrets.get('ALPHA_VANTAGE_API_KEY', '')
AUTHORIZED_USERS = st.secrets.get('AUTHORIZED_USERS', {}) or {'kibe5067@gmail.com': 'mikey19nk'}

ALPACA_API_KEY = st.secrets.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = st.secrets.get('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = st.secrets.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# ---- Auth (model9_1-style) ----
def authenticate_user(email: str, password: str) -> bool:
    return email in AUTHORIZED_USERS and AUTHORIZED_USERS[email] == password

def login_page():
    st.set_page_config(page_title='AI Trading - Login', layout='centered')
    st.title('🤖 AI Trading Agent Management System')
    st.subheader('Secure Login Portal')
    with st.form('login_form'):
        email = st.text_input('📧 Email')
        password = st.text_input('🔑 Password', type='password')
        submitted = st.form_submit_button('🚀 Login')
        if submitted:
            if authenticate_user(email, password):
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.success('✅ Login successful')
                time.sleep(0.5)
                st.rerun()
            else:
                st.error('❌ Invalid email/password')
    with st.expander('Demo Users'):
        for e,p in AUTHORIZED_USERS.items():
            st.write(f'**{e}** / **{p}**')

def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.cache_data.clear()
    st.success('👋 Logged out')
    time.sleep(0.3)
    st.rerun()

def check_authentication() -> bool:
    if not st.session_state.get('authenticated', False):
        login_page(); return False
    return True

# ---- model10 manager ----
@st.cache_resource
def get_manager() -> 'm10.EnhancedManagerAgent':
    return m10.EnhancedManagerAgent()

# ---- Broker client (Alpaca) ----
class AlpacaError(Exception): pass

@dataclass
class OrderResult:
    id: str=''; status: str=''; symbol: str=''; qty: str=''; side: str=''
    type: str=''; time_in_force: str=''; filled_qty: str=''; created_at: str=''

class AlpacaClient:
    def __init__(self, key: str, secret: str, base: str):
        self.base = base.rstrip('/')
        self.s = requests.Session()
        self.s.headers.update({'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret, 'Content-Type': 'application/json'})
    def _ok(self, r):
        try: j = r.json()
        except Exception: r.raise_for_status(); raise AlpacaError('Non-JSON response')
        if not r.ok: raise AlpacaError(j.get('message') or f'HTTP {r.status_code}: {j}')
        return j
    def account(self)->Dict[str,Any]: return self._ok(self.s.get(f'{self.base}/v2/account', timeout=20))
    def positions(self)->List[Dict[str,Any]]: return self._ok(self.s.get(f'{self.base}/v2/positions', timeout=20))
    def place(self, symbol:str, qty:int, side:str, tif:str='day')->OrderResult:
        j = self._ok(self.s.post(f'{self.base}/v2/orders', json={
            'symbol': symbol.upper(), 'qty': str(int(qty)), 'side': side.lower(),
            'type': 'market', 'time_in_force': tif.lower()
        }, timeout=25))
        return OrderResult(**{k:j.get(k,'') for k in OrderResult().__dict__.keys()})
    def cancel(self, oid:str)->Dict[str,Any]:
        r = self.s.delete(f'{self.base}/v2/orders/{oid}', timeout=15)
        return {'status':'canceled','id':oid} if r.status_code==204 else self._ok(r)

# ---- ExecutionAgent ----
class ExecutionAgent:
    def __init__(self, client: Optional[AlpacaClient]): self.client = client
    @property
    def ready(self): return self.client is not None
    def submit(self, symbol:str, action:str, qty:int, tif:str='day')->Dict[str,Any]:
        if not self.ready: return {'ok':False,'error':'Broker not configured'}
        try:
            res = self.client.place(symbol, qty, action, tif); return {'ok':True,'order':res.__dict__}
        except Exception as e: logger.exception('Order failed'); return {'ok':False,'error':str(e)}
    def account(self):
        if not self.ready: return {'ok':False,'error':'Broker not configured'}
        try: return {'ok':True,'account': self.client.account()}
        except Exception as e: return {'ok':False,'error':str(e)}
    def positions(self):
        if not self.ready: return {'ok':False,'error':'Broker not configured'}
        try: return {'ok':True,'positions': self.client.positions()}
        except Exception as e: return {'ok':False,'error':str(e)}
    def cancel(self, oid:str):
        if not self.ready: return {'ok':False,'error':'Broker not configured'}
        try: return {'ok':True,'result': self.client.cancel(oid)}
        except Exception as e: return {'ok':False,'error':str(e)}

@st.cache_resource
def get_exec()->ExecutionAgent:
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL): return ExecutionAgent(None)
    return ExecutionAgent(AlpacaClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL))

# ---- App ----
def main():
    if not check_authentication(): return
    st.set_page_config(page_title='AI Trading - Paper Execution', layout='wide', initial_sidebar_state='expanded')

    c1,c2,c3 = st.columns([3,1,1])
    with c1: st.markdown('# 🤖 AI Trading Agent Management System\n**Now with Paper Trading (Alpaca)**')
    with c2: st.markdown(f'**👤 User:** {st.session_state.user_email}\n\n**🔑 Role:** {"Admin" if st.session_state.get("is_admin") else "User"}')
    with c3:
        if st.button('🚪 Logout', use_container_width=True): logout()

    manager = get_manager()
    exec_agent = get_exec()

    with st.sidebar:
        st.markdown('### 🎛️ System Controls')
        try: m10.display_enhanced_system_status(manager)
        except Exception as e: st.warning(f'System status unavailable: {e}')
        st.markdown('---')
        if st.button('🔄 Refresh All Data'): st.cache_data.clear(); st.success('Refreshed'); time.sleep(0.2); st.rerun()
        if st.button('▶️ Execute All Tasks'):
            with st.spinner('Running tasks...'): n = len(manager.execute_all_pending_tasks()); st.success(f'Executed {n} tasks'); st.rerun()
        st.markdown('---')
        st.markdown('### 🧪 Trading Connection')
        st.success('Alpaca Paper: Connected') if exec_agent.ready else st.warning('Alpaca Paper: Not configured')

    tab1, tab2, tab3, tab4 = st.tabs(['📊 Dashboard','📚 History','🛡️ Monitor','💹 Trading'])
    with tab1:
        try: m10.display_enhanced_agent_dashboard(manager)
        except Exception as e: st.error(f'Agent dashboard error: {e}')
    with tab2:
        try: m10.display_enhanced_task_history(manager)
        except Exception: st.info('History UI not available'); st.write(getattr(manager,'task_history',[]))
    with tab3:
        try: m10.display_system_monitor(manager)
        except Exception: st.info('System monitor not available')

    with tab4:
        st.header('💹 Paper Trading (Alpaca)')
        if not exec_agent.ready:
            st.error('Configure ALPACA_API_KEY & ALPACA_SECRET_KEY in st.secrets to enable trading.')
        else:
            st.subheader('Account Overview')
            acc = exec_agent.account()
            if acc.get('ok'):
                a = acc['account']
                c1,c2,c3,c4 = st.columns(4)
                def to_float(x):
                    try: return float(x)
                    except: return 0.0
                c1.metric('Equity', f"${to_float(a.get('equity',0)):.2f}")
                c2.metric('Cash', f"${to_float(a.get('cash',0)):.2f}")
                c3.metric('Buying Power', f"${to_float(a.get('buying_power',0)):.2f}")
                c4.metric('Portfolio Value', f"${to_float(a.get('portfolio_value',0)):.2f}")
                with st.expander('Account (raw)'): st.write(a)
            else:
                st.warning(acc.get('error'))

            st.markdown('---')
            st.subheader('Open Positions')
            pos = exec_agent.positions()
            if pos.get('ok') and pos['positions']:
                rows = [{k:v for k,v in p.items() if k in ['symbol','qty','avg_entry_price','market_value','unrealized_pl','unrealized_plpc']} for p in pos['positions']]
                st.dataframe(rows, use_container_width=True)
                with st.expander('Positions (raw)'): st.write(pos['positions'])
            else:
                st.info('No open positions.')

            st.markdown('---')
            st.subheader('Place Market Order')
            with st.form('order_form'):
                c1,c2,c3,c4 = st.columns([2,1,1,1])
                sym = c1.text_input('Symbol', 'AAPL')
                side = c2.selectbox('Side', ['buy','sell'])
                qty = c3.number_input('Quantity', min_value=1, value=1, step=1)
                tif = c4.selectbox('Time in Force', ['day','gtc'])
                go = st.form_submit_button('Submit Order ✅')
                if go:
                    with st.spinner('Submitting...'):
                        res = exec_agent.submit(sym, side, int(qty), tif)
                        if res.get('ok'):
                            st.success(f"Order {res['order']['status']} | ID: {res['order']['id']}")
                            with st.expander('Order details'): st.write(res['order'])
                        else:
                            st.error(f"Order failed: {res.get('error')}")

    st.markdown('---')
    c1,c2,c3 = st.columns(3)
    with c1: st.write(f"**Gemini AI:** {'🟢 Connected' if GEMINI_API_KEY else '🔴 Not Configured'}")
    with c2: st.write(f"**Alpha Vantage:** {'🟢 Connected' if ALPHA_VANTAGE_API_KEY else '🟡 Optional'}")
    with c3: st.write(f"**Broker (Alpaca):** {'🟢 Paper' if exec_agent.ready else '🔴 Disabled'} | **Last Updated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
