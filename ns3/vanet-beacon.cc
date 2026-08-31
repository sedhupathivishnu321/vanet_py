/*
 * vanet-beacon.cc  --  IEEE 802.11p Basic-Safety-Message (BSM) beaconing over a
 * SUMO/IDM-derived vehicle mobility trace, for the puducherry-vanet-transfer
 * project.  Drop this file into  <ns-3>/scratch/  and build with:
 *
 *     ./ns3 build vanet-beacon
 *
 * Run (invoked by src/vanet/ns3_channel.py):
 *
 *     ./ns3 run "vanet-beacon
 *          --mobility=/abs/path/mobility.tcl --nNodes=120 --duration=1800
 *          --beaconHz=10 --pktBytes=200 --txPowerDbm=20
 *          --propagation=LogDistance --phyMode=OfdmRate6MbpsBW10MHz
 *          --rxTrace=/abs/path/reception.csv --seed=42"
 *
 * Output CSV columns:  event,t,gen_t,node,peer,dist_m
 *   event = tx : a beacon was broadcast              (peer=-1, gen_t=t)
 *   event = rx : a beacon was received               (peer = tx node id,
 *                                                      gen_t = its generation time,
 *                                                      dist_m = tx-rx distance)
 * Python derives realised PDR, latency and Age-of-Information from this trace.
 *
 * Tested against ns-3.40 / 3.41 / 3.42 with modules:
 *   core network internet mobility wifi wave applications propagation stats
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/wave-module.h"
#include "ns3/propagation-module.h"

#include <fstream>
#include <cstring>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("VanetBeacon");

static std::ofstream g_trace;
static uint32_t g_txCount = 0, g_rxCount = 0;

#pragma pack(push, 1)
struct Bsm { uint32_t id; double genT; double x; double y; };
#pragma pack(pop)

static Vector NodePos (Ptr<Node> n)
{
  Ptr<MobilityModel> m = n->GetObject<MobilityModel> ();
  return m ? m->GetPosition () : Vector (0, 0, 0);
}

static void ReceivePacket (Ptr<Socket> socket)
{
  Ptr<Node> rxNode = socket->GetNode ();
  Ptr<Packet> pkt;
  Address from;
  while ((pkt = socket->RecvFrom (from)))
    {
      if (pkt->GetSize () < sizeof (Bsm)) continue;
      Bsm b;
      pkt->CopyData (reinterpret_cast<uint8_t *> (&b), sizeof (Bsm));
      if (b.id == rxNode->GetId ()) continue;             // ignore own broadcast
      Vector rp = NodePos (rxNode);
      double dx = rp.x - b.x, dy = rp.y - b.y;
      double dist = std::sqrt (dx * dx + dy * dy);
      g_trace << "rx," << Simulator::Now ().GetSeconds () << "," << b.genT << ","
              << rxNode->GetId () << "," << b.id << "," << dist << "\n";
      g_rxCount++;
    }
}

static void SendBeacon (Ptr<Socket> socket, uint32_t pktBytes, double periodS,
                        Ptr<UniformRandomVariable> jitter)
{
  Ptr<Node> n = socket->GetNode ();
  Vector p = NodePos (n);
  Bsm b { n->GetId (), Simulator::Now ().GetSeconds (), p.x, p.y };
  uint32_t sz = std::max<uint32_t> (pktBytes, sizeof (Bsm));
  std::vector<uint8_t> buf (sz, 0);
  std::memcpy (buf.data (), &b, sizeof (Bsm));
  socket->Send (Create<Packet> (buf.data (), sz));
  g_trace << "tx," << b.genT << "," << b.genT << "," << n->GetId () << ",-1,0\n";
  g_txCount++;
  Simulator::Schedule (Seconds (periodS + jitter->GetValue (0.0, 0.01)),
                       &SendBeacon, socket, pktBytes, periodS, jitter);
}

int main (int argc, char *argv[])
{
  std::string mobility, rxTrace = "reception.csv";
  std::string phyMode = "OfdmRate6MbpsBW10MHz";
  std::string propagation = "LogDistance";
  uint32_t nNodes = 0, pktBytes = 200;
  double duration = 600.0, beaconHz = 10.0, txPowerDbm = 20.0;
  uint32_t seed = 1;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("mobility", "ns-2 mobility trace file", mobility);
  cmd.AddValue ("nNodes", "number of (equipped) vehicle nodes", nNodes);
  cmd.AddValue ("duration", "simulation duration [s]", duration);
  cmd.AddValue ("beaconHz", "beacon rate [Hz]", beaconHz);
  cmd.AddValue ("pktBytes", "beacon payload size [B]", pktBytes);
  cmd.AddValue ("txPowerDbm", "transmit power [dBm]", txPowerDbm);
  cmd.AddValue ("phyMode", "802.11p PHY data mode", phyMode);
  cmd.AddValue ("propagation", "LogDistance | Nakagami | Friis", propagation);
  cmd.AddValue ("rxTrace", "output CSV path", rxTrace);
  cmd.AddValue ("seed", "RNG run number", seed);
  cmd.Parse (argc, argv);

  NS_ABORT_MSG_IF (mobility.empty () || nNodes == 0, "need --mobility and --nNodes");
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (seed);

  NodeContainer nodes;
  nodes.Create (nNodes);

  // --- mobility from the SUMO/IDM trace --------------------------------------
  Ns2MobilityHelper ns2 (mobility);
  ns2.Install ();

  // --- 802.11p PHY / channel ----------------------------------------------
  YansWifiChannelHelper channel;
  channel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  if (propagation == "Friis")
    channel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
  else if (propagation == "Nakagami")
    {
      channel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                  "Exponent", DoubleValue (2.5));
      channel.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
    }
  else
    channel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                "Exponent", DoubleValue (2.5));

  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());
  phy.Set ("TxPowerStart", DoubleValue (txPowerDbm));
  phy.Set ("TxPowerEnd", DoubleValue (txPowerDbm));

  Wifi80211pHelper wifi80211p = Wifi80211pHelper::Default ();
  NqosWaveMacHelper mac = NqosWaveMacHelper::Default ();
  wifi80211p.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                      "DataMode", StringValue (phyMode),
                                      "ControlMode", StringValue (phyMode));
  NetDeviceContainer devices = wifi80211p.Install (phy, mac, nodes);

  // --- IP + UDP broadcast sockets ---------------------------------------
  InternetStackHelper internet;
  internet.Install (nodes);
  Ipv4AddressHelper addr;
  addr.SetBase ("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer ifs = addr.Assign (devices);

  g_trace.open (rxTrace);
  g_trace << "event,t,gen_t,node,peer,dist_m\n";

  Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable> ();
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  InetSocketAddress bcast (Ipv4Address ("255.255.255.255"), 9999);

  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<Socket> rx = Socket::CreateSocket (nodes.Get (i), tid);
      rx->Bind (InetSocketAddress (Ipv4Address::GetAny (), 9999));
      rx->SetRecvCallback (MakeCallback (&ReceivePacket));

      Ptr<Socket> tx = Socket::CreateSocket (nodes.Get (i), tid);
      tx->SetAllowBroadcast (true);
      tx->Connect (bcast);
      double start = 0.1 + jitter->GetValue (0.0, 1.0 / beaconHz);
      Simulator::Schedule (Seconds (start), &SendBeacon, tx, pktBytes,
                           1.0 / beaconHz, jitter);
    }

  Simulator::Stop (Seconds (duration));
  Simulator::Run ();
  Simulator::Destroy ();
  g_trace.close ();

  std::cout << "vanet-beacon: tx=" << g_txCount << " rx=" << g_rxCount
            << " trace=" << rxTrace << std::endl;
  return 0;
}
