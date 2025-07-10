
package acsse.csc2a.file;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import acsse.csc2a.model.Ship;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
/**
 * 
 *
 */
public class FileHandler {
	private Ship[] arrShip;
	
	public Ship[] readFile(String Filename)
	{
		File textfile = new File(Filename);
		try(Scanner textin = new Scanner(textfile))
		{
			int n = 0;
			while(textin.hasNext())
			{
				
				String line = textin.nextLine();
				Pattern ShipPattern = Pattern.compile("(^[SH]\\d{4})[a-zA-Z]");
				Matcher shipMatcher = ShipPattern.matcher(line);
				if (shipMatcher.matches())
				{
					System.out.println(line + "matches");
				}
				else
				{
					n = n +1;
					System.out.println("hh" + n);
					
				}
			}
		}
		catch(FileNotFoundException ex)
		{
			ex.printStackTrace();
		}
		Ship[] aryShip = null;
		
		
		return aryShip;
	}
}
