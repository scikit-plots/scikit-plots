Example OpenVPN Script
======================

.. include:: urls.rst

Here is an example of how you can use shell scripts to set up an OpenVPN
server.

Setting up an OpenVPN server using Docker consists of five commands. We can
wrap these commands in a script. The script completes the process from start
to finish. It has a menu for the user to (1) install it on a new VPS
or rebuild, (2) remove and restart the containers, and (3) generate
client configs.

.. code-block:: bash

        #!/bin/bash

        ###############################################################################
        # © Copyright 2019, BilimEdtech | CC BY 4.0 |
        # This script automates building a Docker-based OpenVPN server from
        # https://github.com/kylemanna/docker-openvpn
        #
        # kylemanna's docker-openvpn project has 5 basic commands. This script
        # automates these five commands to set up an OpenVPN server with additional
        # configurations:
        #  - multiple ports
        #  - a stronger cipher
        #  - TLS auth using 'auth SHA256'
        #
        # 1) Generate the openvpn.conf file.
        #    docker run -v $PWD/vpn-data:/etc/openvpn --rm myownvpn ovpn_genconfig -u udp://IP_ADDRESS:1194
        #
        # 2) Init the public key infrastructure (PKI).
        #    docker run -v $PWD/vpn-data:/etc/openvpn --rm -it myownvpn ovpn_initpki
        #
        # 3) Run the OpenVPN server.
        #    docker run -v $PWD/vpn-data:/etc/openvpn -d -p 1194:1194/udp --restart unless-stopped --cap-add=NET_ADMIN myownvpn
        #
        # 4) Generate client config.
        #    docker run -v $PWD/vpn-data:/etc/openvpn --rm -it myownvpn easyrsa build-client-full user1 nopass
        #
        # 5) Retrieve the config from the docker container
        #    docker run -v $PWD/vpn-data:/etc/openvpn --rm myownvpn ovpn_getclient user1 > user1.ovpn
        #
        ###############################################################################

        # Uses the /opt folder for the VPN volume data and configs

        ###----- Variables
        DIRECTORY="/opt/openvpn-udp"
        IMAGE="openvpn-local"
        CONTAINER="openvpn-udp"
        IP_ADDRESS=$(hostname -I | cut -d' ' -f1);
        DEFAULT_PORT=5004
        UDP_PORTS="80 123 443 972 1234 1935 3074 3748 5004 5730 8080 11111 12345"
        CLIENT="client1"
        CIPHER="\n
        # Uses a stronger cipher for a more secure connection \n
        cipher AES-128-CBC \n
        tls-version-min 1.2 \n
        tls-cipher TLS-DHE-RSA-WITH-AES-128-GCM-SHA256 \n
        auth SHA256";
        ###----- End Variables

        # Create $DIRECTORY if it does not exist
        if [ ! -d $DIRECTORY ]; then
            mkdir -p $DIRECTORY
        fi

        #-----------------------------------------------------------
        # - Clones the GitHub site
        # - Builds a new Docker image
        # - Sets up the server config and keys
        #-----------------------------------------------------------
        create_new_instance () {

            read -r -p "Are you sure you want completely remove and rebuild OpenVPN? ?[N/y] " RESPONSE
            if [[ ! $RESPONSE =~ ^[Yy]$ ]]
            then
                exit;
            fi

            # TODO: Check if exists before removing to prevent displaying error messages
            docker stop $CONTAINER && docker rm $CONTAINER
            rm -r $DIRECTORY/vpn-data
            rm -r $DIRECTORY/docker-openvpn
            rm $DIRECTORY/*.ovpn

            # Build the image from Dockerfile
            cd $DIRECTORY
            git clone https://github.com/kylemanna/docker-openvpn.git

            cd $DIRECTORY/docker-openvpn/
            docker build -t $IMAGE .

            echo ""
            echo "-----------------------------------------------------------"
            echo "Setting up environment (Generating configs and keys)"
            echo "-----------------------------------------------------------"
            echo "running command:"
            echo "docker run -v $DIRECTORY/vpn-data:/etc/openvpn --rm $IMAGE ovpn_genconfig -u udp://$IP_ADDRESS:$DEFAULT_PORT"
            echo ""

            docker run -v $DIRECTORY/vpn-data:/etc/openvpn --rm $IMAGE ovpn_genconfig -u udp://$IP_ADDRESS:$DEFAULT_PORT
            docker run -v $DIRECTORY/vpn-data:/etc/openvpn --rm -it $IMAGE ovpn_initpki

            ## Add stronger cipher to the server config
            echo -e $CIPHER >> $DIRECTORY/vpn-data/openvpn.conf
            echo "tls-server" >> $DIRECTORY/vpn-data/openvpn.conf
        }

        #-----------------------------------------------------------
        # Opens the firewall for each port in the list
        #-----------------------------------------------------------
        open_ports () {

            for value in $UDP_PORTS
            do
                echo "Adding rule: ufw allow $value/udp"
                ufw allow $value/udp
            done

            # Send 'y' to enable the new rules
            yes | ufw enable
        }

        #-----------------------------------------------------------
        # - Opens the listed ports in the firewall
        # - Restarts the container to add ports
        # - Restarts the container to apply config changes
        #-----------------------------------------------------------
        restart_container () {

            echo ""
            echo "-----------------------------------------------------------"
            echo "Attempting to remove the existing container named '$CONTAINER'"
            echo "-----------------------------------------------------------"
            echo ""

            echo "Opening firewall for selected ports"
            open_ports

            docker stop $CONTAINER && docker rm $CONTAINER

            # Build the port list for the container
            for value in $UDP_PORTS
            do
                PORT_LIST="$PORT_LIST -p $value:1194/udp"
            done

            echo "Executing docker run --name $CONTAINER -v $DIRECTORY/vpn-data:/etc/openvpn -d $PORT_LIST --restart unless-stopped --cap-add=NET_ADMIN $IMAGE"
            docker run --name $CONTAINER -v $DIRECTORY/vpn-data:/etc/openvpn -d $PORT_LIST --restart unless-stopped --cap-add=NET_ADMIN $IMAGE
        }

        #-----------------------------------------------------------
        # - Prompts the user for name of the client config
        # - Generates the new config
        # - Extracts the client config from the container
        #-----------------------------------------------------------
        create_client_config () {

            echo .
            echo "-----------------------------------------------------------"
            echo "Building client config named '$CLIENT.ovpn'"
            echo "-----------------------------------------------------------"
            echo ""

            read -r -p "Please enter name for client [client1]: " RESPONSE
            if [ -n "$RESPONSE" ]; then
                CLIENT=$RESPONSE
            fi

            docker run -v $DIRECTORY/vpn-data:/etc/openvpn --rm -it $IMAGE easyrsa build-client-full $CLIENT nopass

            echo ""
            echo "Extracting key..."
            docker run -v $DIRECTORY/vpn-data:/etc/openvpn --rm $IMAGE ovpn_getclient $CLIENT > $DIRECTORY/$CLIENT.ovpn

        # Fix comp-lzo config issue
            echo "" >> $DIRECTORY/$CLIENT.ovpn
            echo "comp-lzo no" >> $DIRECTORY/$CLIENT.ovpn

            # Add cipher
            echo -e $CIPHER >> $DIRECTORY/$CLIENT.ovpn
            echo "tls-client" >> $DIRECTORY/$CLIENT.ovpn

            # Add port list for reference
            echo "" >> $CLIENT.ovpn
            echo "# Ports available" >> $DIRECTORY/$CLIENT.ovpn
            echo "# $UDP_PORTS"  >> $DIRECTORY/$CLIENT.ovpn

            echo ""
            echo "Please download '$DIRECTORY/$CLIENT.ovpn'"
            echo "-----------------------------------------------------------"
            exit;
        }

        while :
        do
            echo ""
            cat<<EOF
        ==============================
        OpenVPN Init
        ------------------------------
        Please enter your choice:

        (1) Set up OpenVPN (rebuild)
        (2) Restart container
        (3) Create client config
        (Q)uit
        ------------------------------
        EOF
            read -n1 -s
            case "$REPLY" in
            "1")  create_new_instance
                restart_container
                echo ""
                create_client_config
                echo ""
                exit
            ;;
            "2")  restart_container
                echo ""
                exit
            ;;
            "3")  create_client_config
                echo ""
                exit
            ;;
            "Q")  echo ""
                exit
            ;;
            "q")  echo ""
                exit
            ;;
            * )  echo "invalid option"     ;;
            esac
            sleep 1
        done

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/cloud-computing/7-scripting/openvpn-script.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
