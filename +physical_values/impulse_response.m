%
% superclass for all impulse responses
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-02-03
%
classdef impulse_response < physical_values.signal

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = impulse_response( h_tilde, f_s )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.signal( h_tilde, f_s );

        end % function objects = impulse_response( h_tilde, f_s )

    end % methods

end
