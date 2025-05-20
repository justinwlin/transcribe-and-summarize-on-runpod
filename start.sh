#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #

# Start Ollama service
start_ollama() {
    echo "Starting Ollama service..."
    nohup ollama serve > /ollama.log 2>&1 &
    
    # Wait for Ollama to start up
    echo "Waiting for Ollama to start..."
    until curl -s http://localhost:11434/api/version >/dev/null 2>&1; do
        sleep 1
    done
    
    # Pull the model
    echo "Pulling deepseek-r1:1.5b model..."
    ollama pull deepseek-r1:1.5b
    
    # Run the model in the background
    echo "Starting deepseek-r1:1.5b model..."
    nohup ollama run deepseek-r1:1.5b > /deepseek.log 2>&1 &
    
    echo "Ollama service started with deepseek-r1:1.5b model"
}

# Start nginx service
start_nginx() {
    echo "Starting Nginx service..."
    service nginx start
}

# Execute script if exists
execute_script() {
    local script_path=$1
    local script_msg=$2
    if [[ -f ${script_path} ]]; then
        echo "${script_msg}"
        bash ${script_path}
    fi
}

# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        # Generate SSH host keys if not present
        generate_ssh_keys
        service ssh start
        echo "SSH host keys:"
        cat /etc/ssh/*.pub
    fi
}

# Generate SSH host keys
generate_ssh_keys() {
    ssh-keygen -A
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# Start jupyter lab
start_jupyter() {
    echo "Starting Jupyter Lab..."
    mkdir -p /workspace && \
    cd / && \
    nohup jupyter lab --allow-root --no-browser --port=8888 --ip=* --NotebookApp.token='' --NotebookApp.password='' --FileContentsManager.delete_to_trash=False --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace &> /jupyter.log &
    echo "Jupyter Lab started without a password"
}

# Call Python handler if mode is serverless or both
call_python_handler() {
    echo "Calling Python handler.py..."
    python rp_handler.py
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

start_nginx

echo "Pod Started"

setup_ssh

# start ollama server
start_ollama
# Check MODE_TO_RUN and call functions accordingly
case $MODE_TO_RUN in
    serverless)
        echo "Running serverless mode"
        call_python_handler
        ;;
    pod)
        echo "Running pod mode"
        # Pod mode implies only starting services without calling handler.py
        start_jupyter
        ;;
    *)
        echo "Invalid MODE_TO_RUN value: $MODE_TO_RUN. Expected 'serverless', 'pod', or 'both'."
        exit 1
        ;;
esac

# Confirming ollama is running by running ollama ps
ollama ps

export_env_vars

echo "Start script(s) finished, pod is ready to use."

sleep infinity