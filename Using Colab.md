## Clone a private GitHub repo safely in Python

	import subprocess
	from getpass import getpass
	import urllib

	def git_clone(repo_name):
		try:
			child = subprocess.Popen(
				["git", 'clone', "https://{0}:{1}@github.com/{2}.git".format(
							urllib.parse.quote(input('Username: ')), 
							urllib.parse.quote(getpass('Password/Access Token: ')), 
							repo_name )
				], 
				stdout=subprocess.PIPE, 
				stderr=subprocess.STDOUT
			)
			child.wait()

			if child.returncode == 0:
				# print(out.communicate()[0])
				print(child.stdout.read().decode("utf-8"))
				return 0
			else:
				print("An error occured")
				return 1
		except:
			print("An error occured")
			return 1
