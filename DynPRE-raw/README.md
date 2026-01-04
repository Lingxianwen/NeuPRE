# DynPRE 
## Install



```shell
apt-get update
apt-get install -y python3 python3-dev python3-setuptools virtualenv build-essential libpcap-dev libgraph-easy-perl libffi-dev
pip3 install -r requirements.txt
```



## Tool Usage

Use Modbus as an example. The *server under learning* is selected from a widely-used project [libmodbus](https://github.com/stephane/libmodbus.git)

0. Build the server and prepare traffic 
  
   * Build the server:
   ```shell
   git submodule update --init --recursive
   cd examples/libmodbus
   ./autogen.sh && ./configure
   make
   ```

   * Capture traffic by executing the test server and client (Optional. I've already prepared the traffic and put it in the `examples/in-modbus-pcaps` folder):
     * Capture traffic: `tcpdump -i lo -w examples/in-modbus-pcaps/libmodbus-bandwidth_server-rand_client.pcap`
     * Run the server: `examples/libmodbus/tests/bandwidth-server-many-up`
     * Run the client: `examples/libmodbus/tests/random-test-client`


1. Start the *protocol server under learning*: `examples/libmodbus/tests/bandwidth-server-many-up`.

2. Start reverse engineering: 

   ```shell
   python3 ./DynPRE/DynPRE.py -i ./examples/in-modbus-pcaps -o ./examples/out-modbus -s 10
   ```
   Arguments:
   * `-i`: input folder (pcap files or txt files)
   * `-o`: output folder
   * `-s`: the number of dataset size to be analyzed
   * `-t`: input file message is in text format instead of binary format

3. Then the output is stored in `./examples/out-modbus`. The output file description:

   1. `Segmentation.csv` / `ResponseSegmentation.csv`: field identification result of the response/request messages without refinement

   2. `Segmentation.csv` / `ResponseSegmentation_refine.csv`: field identification result of the request/response messages after refinement 
   
   3. `RefineFields.txt`: common fields identified by the refinement module
   
   4. `RewriteRulesRecord.json`: derived rewrite rules (since Modbus does not employ session-specific identifiers, the rewrite rules is empty. SMB2 employs session-specific identifiers and `examples/out-smb2/RewriteRulesRecord.json` shows the rewrite rules for SMB2 traces recorded in `examples/in-smb2-pcaps`)
   
4. Compare the results with tshark
   1. `apt install tshark`
   2. `pip3 install pyshark`
   3. `python3 ./DynPRE/Evaluator.py ./examples/in-modbus-pcaps '' 3 ./examples/out-modbus/`
   4. `*.csv.perf` : the results of metrics compared to tshark. Especially, `all.perf.csv` restores the overall results. 
   
   