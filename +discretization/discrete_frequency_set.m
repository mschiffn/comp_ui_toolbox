%
% superclass for all discrete frequency sets
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-01-16
%
classdef discrete_frequency_set

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        F_BP ( 1, : ) physical_values.frequency	% set consists of multiple frequencies
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = discrete_frequency_set( interval_f, T_rec )

            if nargin == 0
                return;
            end

            % create column vector of objects
            N_intervals = size( interval_f, 1 );
            obj( N_intervals, 1 ) = obj;

            % check and set independent properties
            for index_interval = 1:N_intervals

                % compute lower and upper bounds on the frequency index
                l_lb = ceil( T_rec( index_interval ) * double( interval_f( index_interval ).bounds( 1 ) ) );
                l_ub = floor( T_rec( index_interval ) * double( interval_f( index_interval ).bounds( 2 ) ) );

                % compute discrete frequencies
                obj( index_interval ).F_BP = physical_values.frequency( (l_lb:l_ub) / T_rec( index_interval ) );
            end
        end
	end % methods

end % classdef discrete_frequency_set
