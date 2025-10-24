import java.util.Scanner;

public class PechayGrowthChecker {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

        System.out.println("=====================================================");
        System.out.println("   🌱 PECHAY GROWTH CHECKER - AUTO GROWTH SIMULATOR");
        System.out.println("=====================================================");
        System.out.print("Enter number of days to monitor (max 365): ");
        int days = input.nextInt();

        if (days < 1 || days > 365) {
            System.out.println("Invalid number of days! Please enter between 1 and 365.");
            return;
        }

        double[] height = new double[days];
        double[] leafSize = new double[days];
        int[] leaves = new int[days];

        // starting values
        height[0] = 1.8;   // cm
        leafSize[0] = 1.2; // cm²
        leaves[0] = 2;     // leaves

        // simulate automatic growth
        for (int i = 1; i < days; i++) {
            height[i] = height[i - 1] + Math.random() * 0.4; // 0.0 - 0.4 cm per day
            leafSize[i] = leafSize[i - 1] + Math.random() * 0.6; // 0.0 - 0.6 cm² per day
            leaves[i] = leaves[i - 1];

            // add new leaf every 20 days (max 50)
            if (i % 20 == 0 && leaves[i] < 50) {
                leaves[i]++;
            }
        }

        // display results
        System.out.println("\n=====================================================");
        System.out.println("      🌿 PECHAY GROWTH OBSERVATION RESULTS");
        System.out.println("=====================================================");
        System.out.println("Day\tHeight(cm)\tLeaf Size(cm²)\tNo. of Leaves");
        System.out.println("-----------------------------------------------------");

        for (int i = 0; i < days; i++) {
            System.out.printf("%d\t%.2f\t\t%.2f\t\t%d\n", (i + 1), height[i], leafSize[i], leaves[i]);
        }

        // calculate summary
        double totalGrowth = height[days - 1] - height[0];
        double avgHeight = 0, avgLeafSize = 0;
        int totalLeaves = 0;

        for (int i = 0; i < days; i++) {
            avgHeight += height[i];
            avgLeafSize += leafSize[i];
            totalLeaves += leaves[i];
        }

        avgHeight /= days;
        avgLeafSize /= days;
        double avgLeaves = (double) totalLeaves / days;

        // display summary
        System.out.println("-----------------------------------------------------");
        System.out.printf("\nInitial Height: %.2f cm\n", height[0]);
        System.out.printf("Final Height (Day %d): %.2f cm\n", days, height[days - 1]);
        System.out.printf("Total Growth: %.2f cm\n", totalGrowth);
        System.out.printf("Average Height: %.2f cm\n", avgHeight);
        System.out.printf("Average Leaf Size: %.2f cm²\n", avgLeafSize);
        System.out.printf("Average No. of Leaves: %.1f\n", avgLeaves);
        System.out.println("-----------------------------------------------------");
        System.out.println("Status: The Pechay plant shows healthy and consistent growth!");
        System.out.println("=====================================================");

        input.close();
    }
}