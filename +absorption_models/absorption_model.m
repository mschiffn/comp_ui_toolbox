%
% superclass for all absorption models
% author: Martin Schiffner
% date: 2016-08-25
%
classdef absorption_model

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        str_name        % name of absorption model
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function ABS = absorption_model( str_name )

            % internal properties
            ABS.str_name	= str_name;
        end

        %------------------------------------------------------------------
        % compute complex-valued wavenumbers
        %------------------------------------------------------------------
        function axis_k_tilde = compute_wavenumbers( ABS, axis_f )

        end

    end % methods

end % classdef absorption_model