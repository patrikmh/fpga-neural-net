#include "Vbroad_core.h"
#include "verilated.h"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        Verilated::commandArgs(argc, argv);
        if (argc != 3) throw std::runtime_error("usage: replay VECTORS COUNT");
        const unsigned count = std::stoul(argv[2]);
        std::ifstream stream(argv[1]);
        if (!stream) throw std::runtime_error("cannot open vectors");
        Vbroad_core dut;
        auto tick = [&]() {dut.clk=0;dut.eval();dut.clk=1;dut.eval();};
        auto reset = [&]() {
            dut.start=0;dut.reset=1;tick();tick();dut.reset=0;tick();
            if(dut.busy || dut.done || dut.output_values || dut.cycles) throw std::runtime_error("reset control/output mismatch");
            for(unsigned j=0;j<48;++j) if(dut.hidden_values[j]) throw std::runtime_error("reset hidden mismatch");
        };
        auto step = [&](const std::vector<uint16_t>& words, unsigned index) {
            for(unsigned j=0;j<5;++j) dut.input_values[j]=0;
            for(unsigned j=0;j<9;++j) dut.input_values[j/2] |= uint32_t(words[j]) << (16*(j%2));
            dut.start=1;tick();dut.start=0;
            if(!dut.busy || dut.done) throw std::runtime_error("start not accepted");
            unsigned observed=1;
            while(!dut.done && observed<1000000) {tick();++observed;}
            if(!dut.done || dut.busy) throw std::runtime_error("completion timeout/busy");
            if(dut.cycles!=observed) throw std::runtime_error("reported cycle mismatch");
            for(unsigned j=0;j<96;++j) {
                uint16_t got=(dut.hidden_values[j/2] >> (16*(j%2))) & 65535;
                if(got!=words[9+j]) throw std::runtime_error("hidden mismatch at step "+std::to_string(index)+" neuron "+std::to_string(j));
            }
            for(unsigned j=0;j<4;++j) {
                uint16_t got=(dut.output_values >> (16*j)) & 65535;
                if(got!=words[105+j]) throw std::runtime_error("raw mismatch at step "+std::to_string(index)+" output "+std::to_string(j));
            }
            tick();
            if(dut.done || dut.busy) throw std::runtime_error("done not a one-clock pulse");
            return observed;
        };
        reset();
        std::string line;
        std::vector<uint16_t> first;
        unsigned latency=0;
        auto started=std::chrono::steady_clock::now();
        for(unsigned i=0;i<count;++i) {
            if(!std::getline(stream,line) || line.size()!=436) throw std::runtime_error("invalid/missing vector row");
            std::vector<uint16_t> words;
            for(unsigned j=0;j<109;++j) words.push_back(std::stoul(line.substr(4*j,4),nullptr,16));
            if(i==0) first=words;
            unsigned observed=step(words,i);
            if(i==0) latency=observed;
            if(observed!=latency) throw std::runtime_error("nonconstant cycle count");
            if((i+1)%1000==0) std::cout << "matched " << i+1 << " continuous steps; cycles=" << latency << std::endl;
        }
        reset();step(first,0);dut.final();
        auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cout << "RESULT {\"matched_steps\":" << count << ",\"cycles\":" << latency << ",\"reset_replay\":true,\"simulation_seconds\":" << seconds << "}" << std::endl;
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "REPLAY_FAILED: " << error.what() << std::endl;
        return 1;
    }
}
