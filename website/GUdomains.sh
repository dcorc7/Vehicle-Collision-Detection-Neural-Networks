

echo "Cleaning up existing build..."
rm -rf Vehicle-Collision-Detection-Project

# build website
echo "Building the website..."
quarto render

# Set Correct File Permissions
echo "Setting correct file permissions..."

find Vehicle-Collision-Detection-Project -type f -exec chmod 644 {} \;

find Vehicle-Collision-Detection-Project -type d -exec chmod 755 {} \;

# push to the Website
read -p "Do you want to push the website to your GU domains folder? (y/n): " answer

if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "Pushing the website to the remote server..."
    scp -r Vehicle-Collision-Detection-Project corcoran@corcoran.georgetown.domains:/home/corcoran/public_html/
    echo "Website successfully pushed!"
else
    echo "Skipping deployment."
fi