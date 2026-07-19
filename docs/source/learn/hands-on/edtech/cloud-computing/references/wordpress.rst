*************************
WordPress Hacks
*************************

.. contents:: Table of Contents

This page contains hacks for working with WordPress.

.. _wp_reset_password:

Reset admin password
=====================

There are two easy ways to reset the password using WordPress in
Docker without accessing the database directly.

These instructions are modified from |Resetting Your Password|
to work with Docker.

Using WP CLI
-------------------------------------
|WP-CLI| is the command-line interface for administrating WordPress
without using a web browser. These instructions should work in most instances.

.. tip::
    You can easily change the password for any user using this method.

#. Install the WP-CLI Phar package

   .. code-block:: bash

       # Change to the docker-compose wordpress directory
       cd ~/wordpress-docker

       # Install wp-cli.phar
       docker-compose exec wordpress curl -O https://raw.githubusercontent.com/wp-cli/builds/gh-pages/phar/wp-cli.phar
       docker-compose exec wordpress chmod +x wp-cli.phar
       docker-compose exec wordpress mv wp-cli.phar /usr/local/bin/wp

   .. code-block:: bash
       :caption: Example output

       root@vps298933:~# cd ~/wordpress-docker
       root@vps298933:~/wordpress-docker# docker-compose exec wordpress curl -O https://raw.githubusercontent.com/wp-cli/builds/gh-pages/phar/wp-cli.phar
         % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                        Dload  Upload   Total   Spent    Left  Speed
       100 5437k  100 5437k    0     0  28.3M      0 --:--:-- --:--:-- --:--:-- 28.2M
       root@vps298933:~/wordpress-docker# docker-compose exec wordpress chmod +x wp-cli.phar
       root@vps298933:~/wordpress-docker# docker-compose exec wordpress mv wp-cli.phar /usr/local/bin/wp
       root@vps298933:~/wordpress-docker#


#. Verify the installation

   .. code-block:: bash

       docker-compose exec wordpress wp --info

   .. code-block:: bash
       :caption: Example output

       root@vps298933:~/wordpress-docker# docker-compose exec wordpress wp --info
       OS:     Linux 4.15.0-124-generic #127-Ubuntu SMP Fri Nov 6 10:54:43 UTC 2020 x86_64
       Shell:
       PHP binary:     /usr/local/bin/php
       PHP version:    7.4.13
       php.ini used:
       WP-CLI root dir:        phar://wp-cli.phar/vendor/wp-cli/wp-cli
       WP-CLI vendor dir:      phar://wp-cli.phar/vendor
       WP_CLI phar path:       /var/www/html
       WP-CLI packages dir:
       WP-CLI global config:
       WP-CLI project config:
       WP-CLI version: 2.4.0
       root@vps298933:~/wordpress-docker#

#. Find the number of the user that you want to change the password for.

   .. code-block:: bash

       docker-compose exec wordpress wp user list --allow-root

   .. code-block:: bash
       :caption: Example output

       root@vps298933:~/wordpress-docker# docker-compose exec wordpress wp user list --allow-root
       +----+------------+--------------+---------------+---------------------+---------------+
       | ID | user_login | display_name | user_email    | user_registered     | roles         |
       +----+------------+--------------+---------------+---------------------+---------------+
       | 1  | test-user  | test-user    | user@host.com | 2020-11-14 04:36:15 | administrator |
       +----+------------+--------------+---------------+---------------------+---------------+
       root@vps298933:

#. Change the password using the new password and user number.

   .. code-block:: bash

       docker-compose exec wordpress wp user update 1 --user_pass=ESzPHYbLkN! --allow-root

   .. code-block:: bash
       :caption: Example output

       root@vps298933:~/wordpress-docker# docker-compose exec wordpress wp user update 1 --user_pass=ESzPHYbLkN! --allow-root
       sh: 1: /usr/sbin/sendmail: not found
       Success: Updated user 1.
       root@vps298933:

Using function ``wp_set_password()``
-------------------------------------
#. Edit the ``functions.php`` file

   .. note:: Verify that ``twentytwenty`` is the latest theme. These
       instructions were written in 2020. 😉

   .. code-block:: bash

       cd ~/wordpress-docker
       nano wordpress/wp-content/themes/twentytwenty/functions.php

   a. Put the ``wp_set_password()`` function on a blank line
      below ``<?php`` to reset the password for the admin user.

      .. code-block:: php

          // Changes the password for the first registered user during WP installation
          wp_set_password( 'ChangeMe!', 1 );

      .. code-block:: php
          :caption: Example

          <?php

          // Changes the password for the first registered user during WP installation
          wp_set_password( 'ChangeMe!', 1 );

          /**
           * Twenty Twenty functions and definitions
           *


   b. Save the file (you can leave it open in the terminal window).

#. Try logging into the admin panel of your website at
   ``http://blog.mysite.com/wp-admin/`` using the password in the file.

   .. note:: The login will fail but will not generate an error.

#. Edit ``functions.php`` again and comment out the command.

   .. code-block:: php

       // Changes the password for the first registered using when WP was set up
       // wp_set_password( 'ChangeMe!', 1 );

   a. Save and exit

#. Log in using the new password


.. |Resetting Your Password| raw:: html

   <a href="https://wordpress.org/support/article/resetting-your-password/" target="_blank">Resetting Your Password</a>

.. |WP-CLI| raw:: html

   <a href="https://wp-cli.org/" target="_blank">WP-CLI</a>

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/cloud-computing/references/wordpress.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
