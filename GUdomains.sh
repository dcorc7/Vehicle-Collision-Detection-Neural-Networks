

echo "Cleaning up existing build..."
rm -rf Collision Detection Neural Networks

# build website
echo "Building the website..."
quarto render

# Set Correct File Permissions
echo "Setting correct file permissions..."

find Collision Detection Neural Networks -type f -exec chmod 644 {} \;

find Collision Detection Neural Networks -type d -exec chmod 755 {} \;

# push to the Website
read -p "Do you want to push the website to your GU domains folder? (y/n): " answer

if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "Pushing the website to the remote server..."
    scp -r Collision Detection Neural Networks corcoran@corcoran.georgetown.domains:/home/corcoran/public_html/
    echo "Website successfully pushed!"
else
    echo "Skipping deployment."
fi