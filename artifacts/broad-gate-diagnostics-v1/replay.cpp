#include "Vbroad_top.h"
#include "verilated.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);Vbroad_top d;d.clk=0;d.rst=1;d.uart_rx=1;
 std::vector<unsigned> reply; int phase=-1,wait=0,bit=0,byte=0;unsigned long ticks=0;
 auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();++ticks;
 if(phase==-1){if(!d.uart_tx){phase=0;wait=651;bit=0;byte=0;}}
 else if(--wait==0){if(phase==0){byte|=unsigned(d.uart_tx)<<bit;if(++bit==8){phase=1;}wait=434;}
 else {if(!d.uart_tx)throw std::runtime_error("bad stop");reply.push_back(byte);phase=-1;}}
 };
 auto run=[&](unsigned n){while(n--)tick();};run(300);d.rst=0;run(100);
 for(auto file:{"request-reset.hex","request-step.hex"}){
 reply.clear();std::ifstream f(std::string("build/broad-post-synth-v1/")+file);std::string hex;f>>hex;
 for(unsigned j=0;j<hex.size();j+=2){unsigned v=std::stoul(hex.substr(j,2),nullptr,16);d.uart_rx=0;run(434);for(int b=0;b<8;++b){d.uart_rx=(v>>b)&1;run(434);}d.uart_rx=1;run(434);}
 unsigned max=1200000;while(reply.size()<210 && max--)tick();
 if(reply.size()!=210)throw std::runtime_error("response timeout");
 std::cout<<file<<" ";for(auto v:reply)printf("%02x",v);std::cout<<" ticks="<<ticks<<std::endl;run(500);
 }
 d.final();return 0;
}
