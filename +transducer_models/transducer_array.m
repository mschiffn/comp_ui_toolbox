%
% superclass for all transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-20
% modified: 2019-02-02
%
classdef transducer_array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_elements_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 1 ]	% numbers of elements along each coordinate axis (1)
        str_name = {'Default Array', 'Default Corp.'}                                                   % name of array

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2              % number of dimensions (1)
        N_elements ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 128              % total number of elements (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = transducer_array( N_elements_axis, str_name )

            % check and set independent properties
            obj.N_dimensions = numel( N_elements_axis );
            obj.N_elements_axis = N_elements_axis;
            obj.str_name = str_name;

            % dependent properties
            obj.N_elements = prod( obj.N_elements_axis, 2 );
        end
	end % methods

end % classdef transducer_array
