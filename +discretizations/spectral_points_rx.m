%
% superclass for all spectral discretizations based on pointwise sampling
%
% author: Martin F. Schiffner
% date: 2019-03-08
% modified: 2019-03-11
%
classdef spectral_points_rx < discretizations.spectral_points_base

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points_rx( indices_active, transfer_functions )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spectral_points_base( indices_active, transfer_functions );

        end % function objects = spectral_points_rx( indices_active, transfer_functions )

	end % methods

end % classdef spectral_points < discretizations.spectral
