#!/usr/bin/env python3
"""
IBKR Connection Test Script
Tests connection to Interactive Brokers and validates setup
"""

import asyncio
import sys
from datetime import datetime
from ib_insync import IB, Stock, util
import nest_asyncio
nest_asyncio.apply()

class IBKRTester:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.host = host
        self.port = port  
        self.client_id = client_id
        self.ib = IB()
        self.connected = False

    async def test_connection(self):
        """Test basic connection to IBKR"""
        print("🔌 Testing IBKR Connection...")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Client ID: {self.client_id}")
        
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            print("✅ Connected to IBKR successfully!")
            self.connected = True
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.print_connection_troubleshooting()
            return False

    def print_connection_troubleshooting(self):
        """Print troubleshooting tips"""
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure TWS or IB Gateway is running")
        print("   2. Check API is enabled: Configure → API → Settings")
        print("   3. Verify port 7497 for paper trading (7496 for live)")
        print("   4. Check 'Enable ActiveX and Socket Clients' is checked")
        print("   5. Add '127.0.0.1' to trusted IPs")

    async def test_account_info(self):
        """Test account information retrieval"""
        if not self.connected:
            return False
        
        print("\n👤 Testing Account Info...")
        try:
            # Get managed accounts
            accounts = self.ib.managedAccounts()
            print(f"📊 Managed Accounts: {accounts}")
            
            if not accounts:
                print("❌ No managed accounts found")
                return False
            
            # Check account type
            account = accounts[0]
            if account.startswith('DU'):
                print("✅ Paper Trading Mode Active")
                is_paper = True
            elif account.startswith('U'):
                print("⚠️  LIVE TRADING MODE - BE CAREFUL!")
                is_paper = False
            else:
                print(f"❓ Unknown account type: {account}")
                is_paper = None
            
            # Get account summary
            print("\n💰 Account Summary:")
            summary = self.ib.accountSummary()
            
            key_metrics = ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'GrossPositionValue']
            
            for item in summary:
                if item.tag in key_metrics:
                    value = float(item.value) if item.value.replace('.', '').replace('-', '').isdigit() else item.value
                    if isinstance(value, float):
                        print(f"   {item.tag}: ${value:,.2f}")
                    else:
                        print(f"   {item.tag}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Account info failed: {e}")
            return False

    async def test_market_data(self):
        """Test market data retrieval"""
        if not self.connected:
            return False
        
        print("\n📈 Testing Market Data...")
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in test_symbols:
            try:
                # Create contract
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                
                # Request market data
                ticker = self.ib.reqMktData(contract, '', False, False)
                await asyncio.sleep(2)  # Wait for data
                
                # Check data
                if ticker.last and ticker.last == ticker.last:  # Not NaN
                    print(f"   ✅ {symbol}: ${ticker.last:.2f}")
                elif ticker.close and ticker.close == ticker.close:
                    print(f"   ⏰ {symbol}: ${ticker.close:.2f} (delayed/close)")
                else:
                    print(f"   ❌ {symbol}: No price data")
                
                # Cancel to avoid fees
                self.ib.cancelMktData(contract)
                
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
        
        return True

    async def test_order_permissions(self):
        """Test if we can place orders (dry run)"""
        if not self.connected:
            return False
        
        print("\n📝 Testing Order Permissions...")
        
        try:
            # Check if we have trading permissions
            accounts = self.ib.managedAccounts()
            if not accounts:
                print("❌ No accounts available for trading")
                return False
            
            account = accounts[0]
            
            # Get account summary to check permissions
            summary = self.ib.accountSummary()
            trading_permissions = []
            
            for item in summary:
                if 'tradingpermissions' in item.tag.lower():
                    trading_permissions.append(item.value)
            
            if trading_permissions:
                print(f"   📋 Trading Permissions: {', '.join(trading_permissions)}")
            else:
                print("   ✅ Basic trading permissions available")
            
            # Check buying power
            buying_power = 0
            for item in summary:
                if item.tag == 'BuyingPower':
                    buying_power = float(item.value)
                    break
            
            if buying_power > 0:
                print(f"   💵 Buying Power: ${buying_power:,.2f}")
                print("   ✅ Ready for trading")
            else:
                print("   ⚠️  No buying power available")
            
            return True
            
        except Exception as e:
            print(f"❌ Order permissions test failed: {e}")
            return False

    async def run_full_test(self):
        """Run complete IBKR setup test"""
        print("🚀 IBKR Setup Test")
        print("=" * 50)
        print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test connection
        if not await self.test_connection():
            return False
        
        # Test account info
        if not await self.test_account_info():
            return False
        
        # Test market data
        await self.test_market_data()
        
        # Test order permissions
        await self.test_order_permissions()
        
        # Disconnect
        self.ib.disconnect()
        
        print("\n" + "=" * 50)
        print("✅ IBKR Setup Test Completed!")
        print("🎯 Your bot should now work with IBKR")
        
        return True

def main():
    """Main test function"""
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'live':
            port = 7496
            print("⚠️  Testing LIVE trading connection (port 7496)")
            confirm = input("Are you sure? Type 'yes' to continue: ")
            if confirm.lower() != 'yes':
                print("Cancelled.")
                return
        else:
            try:
                port = int(sys.argv[1])
                print(f"Using custom port: {port}")
            except ValueError:
                print("Invalid port number. Using default 7497")
                port = 7497
    else:
        port = 7497  # Default paper trading port
        print("Using paper trading port 7497")
    
    # Run test
    tester = IBKRTester(port=port)
    
    try:
        success = asyncio.run(tester.run_full_test())
        
        if success:
            print("\n🎉 All tests passed! You're ready to run the trading bot.")
            print("💡 Next steps:")
            print("   1. Run: python ibkr_bot.py")
            print("   2. Monitor the bot's behavior")
            print("   3. Check positions in TWS")
        else:
            print("\n❌ Some tests failed. Please fix issues before running the bot.")
            
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()