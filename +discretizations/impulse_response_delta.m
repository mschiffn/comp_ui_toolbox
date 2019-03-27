%
% superclass for all delta impulse responses
%
% author: Martin F. Schiffner
% date: 2019-03-02
% modified: 2019-03-02
%
classdef impulse_response_delta < physical_values.impulse_response

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = impulse_response_delta( lbs_q, T_s )

            %--------------------------------------------------------------
            % 1.) compute impulse responses
            %--------------------------------------------------------------
            % create regular discrete time sets
            sets_t = discretizations.set_discrete_time_regular( lbs_q, lbs_q, T_s );

            % assign unity samples
            h_tilde = cell( size( sets_t ) );
            for index_object = 1:numel( sets_t )
                h_tilde{ index_object } = physical_values.physical_value( 1 );
            end % for index_object = 1:numel( sets_t )

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.impulse_response( sets_t, h_tilde );

        end % function objects = impulse_response_delta( lbs_q, T_s )

	end % methods

end % classdef impulse_response_delta < physical_values.impulse_response
