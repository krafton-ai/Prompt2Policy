import { redirect } from "next/navigation";

export default function SchedulerJobRedirect({
  params,
}: {
  params: { jobId: string };
}) {
  redirect(`/benchmark/job/${params.jobId}`);
}
