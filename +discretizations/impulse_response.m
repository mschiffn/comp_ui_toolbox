%
% superclass for all impulse responses
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-02-08
%
classdef impulse_response < physical_values.signal

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = impulse_response( sets_t, h_tilde )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.signal( sets_t, h_tilde );

        end % function objects = impulse_response( sets_t, h_tilde )

	end % methods

end
